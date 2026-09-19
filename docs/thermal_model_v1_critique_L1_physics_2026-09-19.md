# Thermal model v1 — critique L1: physics and thermodynamics (2026-09-19)

> **Lens**: physics and thermodynamics ONLY. Determinism, integers, CUDA, the
> digest and test strategy belong to L2/L3 and are deliberately not examined
> here. §2's rulings (R1–R9) and the ACCEPTED GAPs are Erik's decisions and are
> **not** re-litigated; where a finding touches one it is a check of a *number*
> or of a *stated consequence* inside the reason, never of the decision.
>
> **Document under critique**: `docs/thermal_model_v1_design_2026-09-19.md`
>
> **Also read**: `ray_engine_v2_design_v3_2026-09-15.md` §2.3–2.8, §3, §9;
> `ray_engine_v2_scheme_study_2026-09-13/report_p2b.md`;
> `ray_engine_v2_reach_and_papers_2026-09-13.md` §2;
> `cpp/src/temperature_solver.{h,cpp}`, `cpp/src/combustion.cpp`,
> `src/simulation/materials.py`, `config.toml`.
>
> Every number below was computed, not quoted. The arithmetic is reproduced
> inline so it can be re-run.
>
> **Status**: COMPLETE.

---

## Verdict summary

| severity | count |
|---|---|
| **BLOCKING** | 2 |
| **REQUIRED** | 5 |
| **NOTE** | 6 |

**The single most important finding is B1**: after `cool_shift` dies, three of
the ten shipped material rows have **no conduction face at all** (`κ = 0` →
`NO_FACE` from both sides, structurally), and one of those three (`foliage`)
also has `heat_atten = 0`, so it has **no radiative channel either**. A foliage
tile becomes *strictly adiabatic*: combustion can only add energy to it and
nothing in the engine can ever remove any. §3.1's "exactly three routes" is
false for 30 % of the table, and *zero* routes is the truth for one row. The
design carries no invariant and no test that would catch this.

The model's *core* — one currency for both media, `ΔT = E/C`, a conservative
radiative leak with an exact ambient fixed point — is **sound**, and the ~100×
air figure is **correct** (101.6× exactly). What is not sound is the set of
claims about what the *existing* law already computes: solid conduction is not
"at literature κ" (it is ~5·10⁴× faster — B2), the leak's stated derivation is
not the one that produces its value (R2b), and `A_rad` under-emits by 6.2 %.

### Index

| id | severity | what |
|---|---|---|
| **B1** | BLOCKING | `κ = 0` rows lose route 3 structurally; `foliage` loses every route and becomes adiabatic |
| **B2** | BLOCKING | Ledger item 6 is not derivable as stated: the conduction law's absolute rate is a CFL anchor, ~5·10⁴× literature κ |
| **R1a** | REQUIRED | A genuine fourth *physical* channel: out-of-plane **conduction** into the deck, 2–3.5× the modelled radiative leak for metal |
| **R2b** | REQUIRED | `A_rad` excludes floor/ceiling on an inverted argument → every cell under-emits 6.2 % |
| **R2c** | REQUIRED | R5's "6–10 %" is the **0.9** pin; the shipped pin is **0.7**, where it is 26–40 %. Ledger item 2 ("Done") is downstream of item 3 |
| **R4a** | REQUIRED | Ledger item 7: the solid↔gas face is 47–54× literature κ **and** independent of the solid's material |
| **R4b** | REQUIRED | Ledger item 12 omits the radiative fraction χ_rad; with `object_site` routing a burning crate delivers **zero** energy to room air in v1 |
| **N1** | NOTE | Where `c_v` lives — the seam is the option that keeps the face rule meaningful |
| **N2** | NOTE | Buoyancy cannot be emergent in a top-down plane; "convection is emergent" means expansion-driven flow only |
| **N3** | NOTE | Gas energy is advected as `c_v·T`, not enthalpy `c_p·T` — no flow work, so decompression cooling does not exist |
| **N4** | NOTE | `t_amb_q` as a plane cannot express a sub-ambient ambient: `E°` has no bucket below `E°[0]` |
| **N5** | NOTE | Capacity is not state: a burning tile's `C` does not fall as its fuel burns away |
| **N6** | NOTE | `heat_atten` does two jobs (per-cell extinction *and* emissivity); glass's thermal-IR truth contradicts the row |

---

## 1. Loss-channel accounting (§3.1) — is the list of three complete?

### BLOCKING — B1. Three rows have fewer than three routes; one has none.

§3.1: *"A solid loses heat by exactly three routes, all computed."* Checked
against the shipped table (`config.toml`, parsed):

| material | `thermal_mass` | `conductivity` | `heat_atten` | routes after R1 |
|---|---|---|---|---|
| air | 0 (gas) | 0.024 | 0.0 | n/a |
| hull | 32 | 50.0 | 1.0 | 3 |
| wood | 8 | 0.15 | 1.0 | 3 |
| door / door_closed | 8 | 0.3 | 1.0 | 3 |
| steel | 32 | 45.0 | 1.0 | 3 |
| glass | 16 | 1.0 | 0.3 | 3 |
| **furniture** | 8 | **0.0** | 0.5 | **2** |
| **kindling** | 8 | **0.0** | 0.5 | **2** |
| **foliage** | 8 | **0.0** | **0.0** | **0** |

`conductivity = 0` is not a slow face — it is `NO_FACE` in
`materials.py::_build_conduction_tables` (`face[a][b] = no_face` if either κ is
0), and `temperature_solver.cpp` Pass 2 skips a `NO_FACE` face **from both
sides**. So route 3 (conduction into adjacent gas) does not exist for
furniture, kindling or foliage — not weakly, structurally. `config.toml:1627`
says so outright: *"THE dial for a crate: conductivity = 0 means the ambient
decay is [its only loss]"*. That "ambient decay" is `cool_shift`, which R1
deletes.

Foliage is the sharp case. `heat_atten = 0.0` ⇒ `a_i = 0` ⇒ in the sweep's step
`abs_mat = 0` and `emitted = 0`, so `rad_net[i] += 0` every tick. Route 1 is
gone too. Route 2 (`k_leak`) does **not** rescue it: `leaked − ret` is booked
against the *stream*, not against the cell's stored energy — a cell with
`a_i = 0` never puts its own heat into the stream in the first place. Net:
after T4b a foliage tile can be heated (combustion's `object_site` branch writes
`temperature[s]` directly, `combustion.cpp:1107`) and can **never cool**. It
ratchets to `T_MAX_PHYS` and stays there, re-igniting neighbours through
`apply_temperature_ignition` forever.

This fails silently. Nothing looks wrong on screen; the fire simply never goes
out.

*What is missing is an invariant.* Design v3 §2.3 already carries
`heat_atten > 0 ⇒ thermal_mass > 0` ("an absorbing cell the fold ignores is an
uncounted sink"). R1 makes the **converse** load-bearing, and it is nowhere
stated:

> after `cool_shift` is deleted, every thermal solid must have at least one live
> loss channel — `heat_atten > 0` **or** `conductivity > 0`.

T3 may incidentally fix these three rows (real vegetation κ ≈ 0.1–0.5, real
emissivity ≈ 0.95), but the design must not *depend* on that: T3 is a report,
the rows are Erik's, and a row can be edited later by anyone. §6 needs a
verification item — "no thermal solid is adiabatic" — alongside item 6's
sealed-room test.

### REQUIRED — R1a. A genuine fourth *physical* channel: out-of-plane **conduction**.

At a solid tile the floor and ceiling are not only a radiative boundary; a
bulkhead is welded to a deck plate and conducts into it. `k_leak` is radiative
only. Order of magnitude, painted-steel bulkhead tile at ΔT = 100 K (0.333 m
tile, 2.5 m deck):

* out-of-plane **radiation**: `σ(393⁴ − 293⁴) = 934.6 W/m²`, × `A_fc = 2·0.333²
  = 0.2218 m²` × ε 0.9 → **187 W**;
* out-of-plane **conduction** into a 6 mm deck plate, straight-fin estimate
  `q = √(h_tot·κ·t)·P·ΔT` with `h_tot = 20 W/m²K`, `κ = 50 W/mK`, `t = 0.006 m`,
  `P = 4·0.333 = 1.332 m`: `√(20·50·0.006) = 2.449 W/mK` → **326 W** per contact
  surface, **~650 W** counting deck *and* overhead.

So for metal structure near ambient the omitted channel is **2–3.5× the one
that is modelled**. Two consequences:

1. §3.1's "exactly three routes, all computed" is wrong as a *physical*
   enumeration. It is a complete enumeration of the *coded* routes — a different
   and defensible claim, and the one the section should make.
2. The two channels have different **functional form**. A conductive leak is
   linear in ΔT; a radiative leak goes as `(T⁴ − T_amb⁴)`. `cool_shift`
   (`T -= T >> s`, exactly `dT/dt = −T/τ`) had the *linear* form — i.e. it was
   structurally the right stand-in for conduction into the deck and the wrong
   one for radiation. Replacing it with `k_leak` therefore does **not** preserve
   what it was doing for metals: metal structure will cool much faster when hot
   and much slower when barely warm. (§5 quantifies the crossover.)

`k_leak` is per-tile under R2, which is the natural home for such a coefficient
— but a *radiative* coefficient cannot carry a linear law. Either record the
conductive path as an accepted omission, or say that `k_leak` is being asked to
stand for both and state the ΔT range over which that is honest.

### Checked and found sound

* **A solid completely surrounded by other solids is not adiabatic.** Inside an
  opaque block at uniform T, `rad_net ≈ 0` by balance, but the stream inside the
  block equilibrates to the hot blackbody level while `amb_m` stays at ambient,
  so `leaked − ret = k·(i_in − amb_m) > 0` and the block loses out-of-plane at
  every cell. R2's diagnosis ("with `k_leak = 0` a sealed interior room is
  radiatively adiabatic") is correct and the fix works.
* **A hull tile facing vacuum.** Under R4 the sky is ambient and in-grid vacuum
  tiles are transparent (`a = 0`), so the stream crosses them and books to
  `rad_amb` at the ring. No hole, no double count. (That this cools *slower*
  than today's `cool_shift_vacuum` is the first ACCEPTED GAP — not a finding.)
* **No double-counting among the three.** `rad_net` (material), `rad_flux`
  (body) and `rad_amb` (ambient + leak) are disjoint channels of one traversal
  with `Σ ≡ 0`; conduction is a separate pass with an antisymmetric face
  quantum. There is no path by which a joule is charged twice. One nuance worth
  recording in §3.1: route 1's *sink* is not only other solids — a unit standing
  in the stream absorbs into `rad_flux`, which leaves the thermal system
  entirely (units have no temperature state). It is booked, so the identity
  holds, but "a cell hotter than its surroundings emits more than it absorbs" is
  not the whole description of route 1.
* **No unnoticed sink in the code.** I checked the usual suspects: combustion
  only ever *adds* (`heat_saturating_add(&temperature[s], dT)`,
  `combustion.cpp:1108`; the gas branch goes through
  `gas_energy::deposit_railed`) — there is no pyrolysis endotherm and no
  evaporative term; the water solver's boil does not touch solid temperature;
  Pass 0a's wipes only touch non-thermal-solid cells. The rails (`T_MAX_PHYS`,
  `t_low_rail`) and the conduction capacity floor (`e_cond_cap_sum`) do
  destroy/create energy, but each is counted, so verification item 7 ("nothing
  changes except through a booked channel") survives them.

---

## 2. The currency argument and the ~100× figure (§3.2)

### Verified sound: the two capacities really are one unit.

`cpp/src/temperature_solver.h::cell_capacity_q` produces, in the same Q16.16
`cap` unit:

* `thermal_solid`: `cap = 1 << (heat_inv_shift + 16)` — i.e. `C = 2^s =
  thermal_mass`, the *same* divisor Pass 1's `gain = deposit >> shift` uses;
* gas: `cap = (max(N, n_floor)·c_v) >> 16` — i.e. `C = N·c_v`, the *same*
  divisor Pass 1's `ΔT = E_abs/(max(N,floor)·c_v)` uses.

So `min(cap_i, cap_j)` across a solid–gas face is a genuine min of two heat
capacities and `face_energy_q` returns a real joule count. §3.2's claim that
*the form is unified* is correct, and I found no site where the two are mixed at
different scales.

### Verified sound: the ~100× is right — **101.6×**, not a round number.

```
V_tile      = 0.333² · 2.5                 = 0.2772225 m³
C_tile      = 0.7e6 · V_tile               = 194 055.75 J/K   (doc: 194.06 kJ/K ✓)
J_per_count = C_tile / (8 · 65536)         = 0.37013197 J     (doc ✓)
ρc per thermal_mass unit = 0.7e6 / 8       = 87 500 J/(m³·K)  (doc ✓)
air: 1.2 · 718                             = 861.6 J/(m³·K)   (doc: 862 ✓)
87 500 / 861.6                             = 101.56×
```

At ambient `N = 1.0` and `c_v = 1.0` the gas `cap` is `65536`, bit-identical to
a `thermal_mass = 1` solid's `1 << 16` — so the premise ("an air tile behaves as
`thermal_mass = 1`") holds exactly, not approximately. **The headline figure is
correct.** The value `c_v` must take, on this pin, is

```
c_v = 861.6 / 87 500 = 0.0098469          (≈ 1/101.6)
```

`c_v` is the right constant (not `c_p`): the cell is a fixed control volume and
the stored quantity is internal energy. See **N3** for where `c_p` *should* have
appeared and does not.

### REQUIRED — R2c. R5's "6–10 %" is computed on the **0.9** pin; the shipped pin is **0.7**.

This matters because the design calls the 0.7-vs-0.9 pin *"the only ruling
left"* while presenting the quantisation gap as settled. They are the same
decision. Snapping `thermal_mass = 8·ρc/ρc_pin` to the nearest power of two:

| pin `ρc_wood` | steel (3.8 MJ/m³K) wants → gets | glass (2.0) wants → gets |
|---|---|---|
| **0.9 MJ/m³K** | 33.78 → 32 (**−5.3 %**) | 17.78 → 16 (**−10.0 %**) |
| **0.7 MJ/m³K** (shipped) | 43.43 → 32 (**−26.3 %**) | 22.86 → 32 (**+40.0 %**) |

The 0.9 row reproduces R5's *"steel wants 34 … glass wants 17.8 … 6–10 %"*
exactly — that is where those two numbers come from. The ACCEPTED GAP
*"`thermal_mass` quantisation of 6–10 %"* is therefore **true only on the 0.9
branch**; on the shipped 0.7 branch it is 26–40 %, close to the theoretical
worst case for power-of-two snapping (−29 % / +41 %).

Two consequences the design should carry:

1. The pin ruling is **also** the quantisation ruling. That is an argument Erik
   should have in front of him when he makes it, and it is in neither §2 nor §4.
2. `J_per_count ∝ ρc_pin`, so `rad_scale_derived = σ·A_rad·dt/J_per_count` moves
   with the pin: `2.125632e-08` at 0.7 → `1.653270e-08` at 0.9 (before R2b's
   +6.7 %). **Ledger item 2 is marked "Done" but is downstream of item 3, the
   one ruling the document says is still open.** Item 2 is done *conditionally*;
   the table should say so.

*(This checks the arithmetic inside R5's reason and the ACCEPTED GAP's stated
magnitude. The ruling — `thermal_mass` stays a power of two — stands and is not
questioned.)*

### NOTE — N1. Where `c_v` lives is a real fork, and §3.2 frames it correctly.

`gas_energy = N·T_abs` bakes `c_v = 1` into the *representation*, and the #54
closure identity is denominated in it. Putting `c_v ≠ 1` inside the field
rescales every counter by the same factor (harmless to closure, but every stored
`gas_energy` moves); putting it at the seam leaves the field alone but means the
field is `E/c_v`, not energy. The physics does not choose for you — but note
that the *conduction face* already reads `c_v` through `cell_capacity_q`, so the
seam option is the one that keeps today's `min(cap_i, cap_j)` face rule
meaningful without a second conversion.

---

## 3. Does `ΔT = E/C` hold everywhere it is applied?

### Verified sound — including the two places I expected it to fail.

**The solid branch.** `gain = deposit >> heat_inv_shift` with `C = 2^shift` is
`ΔT = E/C` exactly, and it is the *same* `C` that `cell_capacity_q` hands Pass 2
and the ledger. Consistent.

**The gas branch, where `N` varies.** Pass 1 computes

```
E_abs = deposit · min(N, N_AMB)/N_AMB        (absorption ∝ density)
ΔT    = E_abs / (max(N, n_floor_heat) · c_v)
```

I expected the density-proportional absorption to break `E/C`; it does not. For
`n_floor ≤ N ≤ 1` the two factors of `N` cancel exactly — a cell at 1 % of
ambient density absorbs 1 % of the beam *and* has 1 % of the capacity, so
`ΔT = deposit/(N_AMB·c_v)`, which is the physically right answer, not a
coincidence. The comment at `temperature_solver.cpp:369` is correct.

Below `n_floor_heat` the divisor is floored while the real capacity is `N·c_v`;
the difference is counted (`e_cond_cap_sum` / the P-G5 counters), which is the
honest treatment of a numerical floor.

**Pass 2 for gas.** The endpoint divide is *deleted* for accountable gas cells
(arc #54 P-G1b): the four-face sum `de` **is** the energy change, because the
face quantum is `|ΔT|·C` with `C = N·c_v` and the books' capacity is the same
`N`. That is `ΔT = E/C` applied once rather than twice, and it is right.

**The mixing rule.** The gas energy seam moves mass carrying its source's
`T_abs`, so a cell gaining `ΔN` at `T_s` lands at `(N·T + ΔN·T_s)/(N + ΔN)` —
exact for constant `c_v`. Sound.

### NOTE — N2. Buoyancy cannot be emergent in a top-down plane.

§3.1: *"the gas solver advects the energy away. Convection is therefore
emergent, not a dial."* Half of that is true. What is emergent is
**expansion-driven in-plane flow**: heat raises `T`, `p = C·E` raises pressure,
the EOS drives mass out, and the mass carries its `T_abs` with it. That is real
forced convection and it is genuinely emergent.

What cannot be emergent is **buoyancy**, because in a top-down deck plane the
buoyant direction is the axis that does not exist. The hot upper layer, the
ceiling jet and the stratification that dominate real compartment-fire heat
transport are not merely unresolved — they are unrepresentable in this geometry.
That is not a defect of this design, but "convection is emergent" over-claims,
and since §6 item 10's flashover-adjacent assertions are judged against real
compartment-fire intuition, the sentence should be narrowed to "in-plane,
expansion-driven convection is emergent; out-of-plane buoyant transport is not
modelled".

### NOTE — N3. Gas energy is advected as `c_v·T`, not enthalpy `c_p·T`.

For a fixed control volume, `U = N·c_v·T` is the right *stored* quantity — that
part is correct. But the energy **flux** across a cell face for a flowing gas is
`ṁ·h = ṁ·c_p·T`, not `ṁ·c_v·T`: the missing `p·v = R·T` per unit mass is the
flow work. The seam carries `ΔN·T_abs` (CLAUDE.md: "MOVED mass carries its
source's `T_abs`"), so flow work is absent everywhere. Two consequences:

* energy transport by gas flow is low by `γ = c_p/c_v = 1.4`;
* **there is no decompression cooling.** Venting a fixed-volume cell removes `N`
  at the cell's own `T`, leaving `T` unchanged; real adiabatic blowdown gives
  `T ∝ p^((γ−1)/γ)`. The first ACCEPTED GAP offers *"Cold rooms, if wanted, come
  from the gas leaving (decompression cooling)"* as the fallback for R4 — that
  fallback does not exist in the model as built. **This is not a critique of R4**
  (space at room temperature); it is a check of the consolation offered
  alongside it, which should be struck or replaced with "level design".

### NOTE — N4. A per-tile ambient plane cannot express a sub-ambient ambient.

R3's reason is that Erik wants *"room temp in ship, 0 K outside"* to be
*expressible*. The plane alone does not buy that. `amb_m = (E°[0]·w_m) >> 16`,
and `E°` is indexed by game temperature with 0 = the global ambient reference;
design v3 §2.6 fixes `e_bucket_of(T ≤ 0) = 0`, so **the table has no bucket
below ambient** and a cell whose `t_amb_q` is negative still radiates, and still
receives, the 293 K blackbody. A *hotter*-than-reference ambient is expressible;
a colder one is not, until `E°` is extended below its zeroth bucket. Since R4
puts space at room temperature for v1 this is dormant — but §7.2's draft rule
("THE per-tile ambient reference for every thermal boundary") should record the
limit so the next arc does not discover it by filling the plane with 0 K and
seeing nothing change.

### NOTE — N5. Capacity is not state.

`C = thermal_mass` is a material constant. A burning tile loses fuel mass
monotonically but its heat capacity never falls, so the same stored `E` keeps
mapping to the same `T` while the physical object it represents is becoming a
char shell. Combined with R6 (no two-node solid) this is a consistent lumped
model, but it means `ΔT = E/C` is exact in the engine's currency and drifts from
physical truth over a burn. Worth one line in §2's accepted list next to the
`c_p`-is-constant gap, which is the same class of linearisation.

---

## 4. The calibration ledger (§4) — are the derivations the right ones?

### BLOCKING — B2. Item 6 is not derivable the way the table claims.

Item 6: *"Conduction solid↔solid — κ per row — **Derivation** from literature
κ; **test**: a slab's e-fold against the analytic solution."*

The conduction law's **absolute** rate is not a function of κ. It is set by
`SHIFT_AT_REF = 2` against `KAPPA_REF = 50`, whose own config comment says what
it is: *"metal self-rate = 1/4 (fastest stable on a 4-nbr grid)"* — a **CFL
stability anchor**, not a physical one. `materials.py` then buckets every face
as `shift = round(−log2(hm/KAPPA_REF))`, so supplying literature κ per row moves
only the *ratios* between materials. The absolute rate stays where the stability
anchor put it. Measured against the analytic diffusion number `α·dt/Δx²` at
`Δx = 0.333 m`, `dt = 1/24 s`:

| face | engine per-tick relaxation | true `α·dt/Δx²` | engine / true |
|---|---|---|---|
| steel–steel (κ 50, ρc 3.85e6) | `1/2² = 0.25` | 4.88e−6 | **51 231×** |
| wood–wood (κ 0.15, ρc 0.7e6) | `1/2⁸ = 3.91e−3` | 8.05e−8 | **48 514×** |

So the engine's solid conduction is **~5·10⁴× faster than literature κ**, and
"derivation from literature κ" will not change that by a single factor of two.

The two halves of item 6 are mutually unsatisfiable as written:

* if T3 supplies literature κ and the bake constants stay, the slab e-fold test
  fails by 4–5 orders of magnitude;
* if `SHIFT_AT_REF` is moved to pass the test (it needs +15 to +16 shifts), then
  real conduction at this resolution is what physics says it is —
  `Δx²/α = 8 538 s ≈ 2.4 h` for steel, `5.17·10⁵ s ≈ 6 days` for wood — and
  solid–solid conduction disappears from the game as a mechanism.

That second outcome may well be the honest answer (radiation genuinely dominates
solid heat transfer at these ΔT; see §1), but it is a *decision*, and R7 ("real
numbers throughout, nothing tuned") does not by itself make it. The ledger
currently hides the decision inside the word "derivation". **Item 6 must either
name the absolute-rate anchor as a separate ruling, or state up front that T3's
κ row is a ratio-only input.** Building T3/T4b on the table as written produces
one of the two wrong states silently.

### REQUIRED — R4a. Item 7's "the quantised face rate at air's κ" is not what the law does.

Two facts, both computed from `face_shift_table` and `cell_capacity_q`:

1. **It is 47–54× the literature-κ rate.** For a solid–air face the harmonic
   mean is dominated by air, so `s = 10` for *every* shipped solid, and
   `C_min` is always the air side. Energy per tick `= ΔT·C_air/2¹⁰`. With the
   corrected air capacity (`ρc_air·V_tile = 238.9 J/K`) that is
   `5.60·ΔT W`, i.e. an effective **`h = 6.73 W/(m²·K)`** on the
   `0.333 × 2.5 = 0.8325 m²` face. True Fourier conduction across one tile
   width, `κ_hm·A/Δx`, gives `0.10–0.12 W/K` → **`h ≈ 0.12–0.14 W/(m²·K)`**.
   Ratio 47–54×.
2. **It is independent of the solid's material.** Steel (κ 50), wood (0.15),
   door (0.3) and glass (1.0) all give `s = 10` and the same `C_min`, hence the
   identical `5.60 W/K`. "κ per row" buys nothing at a gas face.

Neither is fatal — `h ≈ 6.7 W/(m²·K)` happens to land squarely in the
natural-convection band (5–25), and a boundary-layer-dominated coefficient
*should* be nearly material-independent, so the law is arguably closer to
reality than an honest κ would be at 0.333 m resolution. But then it is a
**convective** coefficient wearing a conduction name, and R1 leans on it. The
ledger should say which it is, and item 7's test should target an `h`, not a κ.
(Note also the coupling to T2: at today's `c_v = 1` the same face gives
`h = 683 W/(m²·K)` — 100× too strong. The `c_v` correction is what brings this
channel into the physical band, which is a second reason it must land *with* the
flip, as T4b already has it.)

### REQUIRED — R4b. Item 12 omits the radiative fraction, and the routing makes it matter.

Item 12 lists *"heat of combustion (J/kg), burn rate, O₂ demand"*. Two
omissions, in increasing order of importance:

* **Pyrolysis endotherm / char.** Wood's heat of gasification is ~1.8 MJ/kg
  against ~16–19 MJ/kg of combustion, and ~20–25 % of the mass chars rather than
  burning in the gas phase. Omitting the endotherm over-delivers ~10 %. Minor,
  but it belongs on the list next to the heat of combustion it modifies.
* **The radiative fraction `χ_rad` — the partition of the heat release.** In a
  real fire ~30 % of the HRR leaves as radiation and ~70 % as buoyant plume
  enthalpy. This engine puts **100 % of it into the burning tile's sensible
  heat** and lets `E°(T)` re-radiate whatever the tile's temperature then
  implies. That is a different partition, and it is the one that decides whether
  a tile reaches flame temperature — which is the entire subject of R6's
  accepted consequence and §6 item 10.

The routing makes this concrete and it is worth stating in §3.1. In
`combustion.cpp:1017`, `object_site` is true whenever the burning cell is a
thermal solid, and that branch deposits the whole parcel into `temperature[s]`;
only a non-solid accountable cell gets `gas_energy`. So for a burning crate in
v1, after the flip:

* combustion heat → entirely into the crate's own 194 kJ/K;
* conduction to air → **none** (κ = 0, B1);
* radiation into the gas → **none** (air's `heat_atten = 0.0`; the sweep's gas
  channel is design v3 §6.3, i.e. P5, explicitly out of scope for this
  document).

**A burning crate therefore delivers zero energy to the room's air in v1.** The
air heats only where a fire sits on a non-solid cell. That is a defensible v1
scope line, but it is a large behavioural fact that §3.1 currently implies the
opposite of ("conduction into air, after which the gas solver advects the energy
away"), and it should be written down before T4b rather than discovered on the
HUMAN-TEST.

### NOTE — N6. Item 9: `heat_atten` is one number doing two different physical jobs.

In the sweep `a_i` is a **per-cell-crossing extinction** (`abs_mat = stream·a`)
*and*, by Kirchhoff, the **emissivity** (`emitted = src·a`). For an opaque solid
(`a = 1`) the two coincide and there is no issue. For `glass = 0.3` they do not:
as extinction, 0.3 per 0.333 m is an absorption coefficient of 1.07 m⁻¹; as
emissivity, 0.3 is simply wrong — **glass is nearly opaque in the thermal IR,
with ε ≈ 0.85–0.95**, and its visual transparency is a different waveband
entirely. T3's brief ("`heat_atten` as emissivity, from literature") will
therefore produce a glass row that is either a bad emitter or a bad window,
depending on which literature number is taken, and the design gives no rule for
choosing. Since the sweep is a *heat* sweep (light is P6, a separate channel),
the thermal-IR value is the right one — say so in the brief.

### The rest of the ledger — checked, and right

* **Item 5 (`ΔT = E/C`)**: correct, and "verified — it is the definition of heat
  capacity" is fair. §3 above confirms it at every application site including
  the varying-`N` gas branch.
* **Item 8 (`k_leak` per tile)**: the *derivation* exists and is good, but the
  table's range **"~0.06–0.10" is not a tolerance — it spans two incompatible
  derivations**, and the low end is ~46 % wrong at r = 12 (see §5). Collapse it
  to the single slab-law value and name the law. The stated test ("a hot tile's
  cooling curve") cannot discriminate between 0.06 and 0.10 on its own; it needs
  the analytic target stated alongside.
* **Item 10 (ignition temperature)**: ~300–350 °C for piloted ignition of
  cellulosics is the right literature band (the shipped rows already sit at
  280–300). Sound.
* **Item 11 (unit heat damage)**: honestly labelled a test — but *"against a
  stated survivability target"* **is** tuning to a named behaviour, which is the
  thing R7 forbids two pages earlier. Not a physics error; an internal
  inconsistency in the document's own posture, worth one sentence acknowledging
  that the damage law is the one deliberately fitted object in the model.

---

## 5. `k_leak` as the replacement for `cool_shift`

### The geometry, checked

`tile_size_m = 0.333`, deck `H = 2.5 m`:

```
A_lat = 4·0.333·2.5   = 3.3300 m²
A_fc  = 2·0.333²      = 0.2218 m²
A_tot                 = 3.5518 m²
A_fc / A_tot = 0.06244        <- the design's "~6 %"   ✓ arithmetic correct
A_fc / A_lat = 0.06660        <- the right comparator for a coefficient that
                                 multiplies the in-plane stream
```

R2's 6 % is arithmetically right. **But it is the wrong quantity**, and the
clause that follows — *"the same ballpark as the design's derived ~0.10"* — is a
false corroboration. The two numbers answer different questions:

* the **area fraction** is an *emission* split: what share of a cell's own
  radiated power leaves through floor and ceiling;
* the **0.10** is a *transport* coefficient: `reach_and_papers` §2 fits
  `exp(−k·r)` to the exact slab dilution `S(r) = a/√(a²+r²)` with
  `a = (H/2)/tile = 3.754` tiles.

`k_leak` multiplies `i_in` — the *stream* — so only the transport object is
dimensionally the right thing. And the two values are not interchangeable.
Computed, `S(r)` against `(1−k)^r`:

| r (tiles) | exact `S(r)` | `k = 0.10` | err | `k = 0.0666` | err |
|---|---|---|---|---|---|
| 1 | 0.9663 | 0.9000 | −6.9 % | 0.9334 | −3.4 % |
| 3 | 0.7812 | 0.7290 | −6.7 % | 0.8132 | +4.1 % |
| 5 | 0.6004 | 0.5905 | −1.6 % | 0.7085 | **+18.0 %** |
| 8 | 0.4248 | 0.4305 | +1.3 % | 0.5762 | **+35.6 %** |
| 12 | 0.2985 | 0.2824 | −5.4 % | 0.4373 | **+46.5 %** |

Max |error| over r = 1…12: **k = 0.10 → 8.2 %**; **k = 0.0666 → 46.5 %**. The
error-minimising constant is k = 0.094 (7.0 %).

**Verdict: 0.10 is the right number and R2's stated reason for it is wrong.** A
reader who takes the area argument seriously — and §4 item 8 invites exactly
that with its "~0.06–0.10" — picks a value ~46 % off at gameplay range. Drop the
area sentence; cite the slab law.

### REQUIRED — R2b. `A_rad` under-emits by 6.2 %, on an inverted argument.

`config.toml`'s `rad_scale_derived` comment sets `A_rad = 4·0.333·2.5 = 3.33 m²`
(lateral faces only) because *"the floor/ceiling faces are the SEPARATE
out-of-plane channel `k_leak` above — folding them in here would double-count
it."* That is backwards. `k_leak` is the slab *dilution* of an isotropic
emitter, and `S(r)` is derived for an emitter radiating its **total** power into
the slab: the out-of-plane share is removed *along the ray*, not withheld at the
source. `k_leak` emits nothing — its only terms are `leaked = i_in·k` and
`ret = amb_m·k`, both functions of the *stream*, neither a function of the
cell's own `T`. So with `k_leak` live the consistent `A_rad` is the full column
surface **3.5518 m²**, and the shipped value under-emits every cell by
`1 − 3.33/3.5518 = 6.2 %`.

Compounded at r = 1 the delivered flux is `0.90 × 0.9376 = 0.8438` against the
truth `0.9663` — **12.7 % low**, where the leak fit alone is 6.9 % low.

Numerically, at the shipped `J_per_count = 0.37013197 J`:
`rad_scale_derived = σ·A_rad·dt/J_per_count` moves `2.125632e-08 → 2.267241e-08`.
(And it moves again with the pin — see R2c.)

### Does a radiative leak reproduce what `cool_shift` was doing?

No, and the difference is large and direction-known. `cool_shift` is
`T -= T >> s`, an exponential relaxation with e-fold `2^s / 24 s`:

| rows | `cool_shift` | e-fold |
|---|---|---|
| hull / steel / door / door_closed / glass | 5 | **1.33 s** |
| wood / furniture / kindling / foliage | 13 | **341 s** |

A 1.33 s e-fold on a 776 kJ/K steel tile is `C/τ = 582 kW` per kelvin of ΔT —
five orders of magnitude beyond any real surface. The dial was never physical,
so R1 is right that it must go; that is not in dispute. What matters for the
*replacement* is that the two laws cross:

* near ambient, radiation linearises to `4σT_amb³ = 5.705 W/(m²·K)`, so the
  out-of-plane channel at `A_fc` is **1.14 W/K** — against `cool_shift = 5`'s
  **582 000 W/K**, and against the omitted deck conduction's ~3.3–6.5 W/K (R1a);
* the radiative channel steepens with ΔT — `σ(T⁴ − T_amb⁴)` is 1.64× its
  linearisation at ΔT = 100 K and **7.7×** at ΔT = 500 K — so the same
  coefficient that is far too weak near ambient is credible at flame
  temperatures.

That is the correct physics and it is a *good* change. But it means the
post-flip behaviour is not "the same cooling under another name". Structure will
sit warm for a very long time and shed hard only when hot. §6 item 10 already
asserts the thick-structure consequence; the *near-ambient* consequence — a
compartment stays warm long after a fire is out, instead of returning to ambient
in seconds — is the one to state, because it is what a player will notice first
and it has no test on the list.

---

## 6. What I checked, and what I did not

**Checked, with arithmetic, and found correct:**

* the currency pin (`V_tile`, `C_tile = 194.06 kJ/K`, `J_per_count = 0.37013197 J`,
  `87 500 J/(m³·K)` per `thermal_mass` unit) — every figure in §3.2 reproduces;
* the ~100× air claim — **101.56×** on the `c_v` basis, and `c_v` is the correct
  constant for a fixed-volume cell;
* that `cell_capacity_q` genuinely produces one unit for both media, so
  `min(cap_i, cap_j)` across a solid–gas face is meaningful;
* `ΔT = E/C` at every application site — solid deposit, gas deposit with varying
  `N` (the `min(N,1)` absorption and the `N` divisor cancel exactly), Pass 2's
  deleted gas divide, and the seam's mixing rule;
* the conservation structure of the leak channel, its exact ambient fixed point,
  and that `rad_net`/`rad_flux`/`rad_amb` are disjoint (no double-count);
* the 6 % floor/ceiling area fraction (6.244 %) and the `S(r)` slab law
  (`a = 3.754` tiles);
* that `k = 0.10` is a good fit to the slab law (8.2 % max over r = 1…12) and
  `k ≈ 0.066` is not (46.5 %);
* that no unnoticed *code* sink exists on the solid side — combustion only adds,
  water's boil does not touch solid `T`, Pass 0a wipes only non-solids, and
  every rail and floor is counted.

**Checked and found wrong** (the findings above): the completeness of §3.1's
three routes (B1, R1a); the derivability of ledger item 6 (B2); the reason for
`A_rad`'s exclusion of the floor/ceiling (R2b); the pin-independence of R5's
6–10 % and of item 2's "Done" (R2c); item 7's description of the solid↔gas face
(R4a); item 12's partition of the heat release (R4b).

**Deliberately not examined** (other lenses, or rulings):

* determinism, integer arithmetic, ingress doors, `/fp:strict`, the digest,
  goldens, CPU↔GPU bit-identity, int64 headroom — L2;
* the patch cut, systems reuse, test dispositions, the single re-baseline — L3;
* §2's rulings R1–R9 and the ACCEPTED GAPs as decisions. Where a number *inside*
  a ruling's reason was checked (R5's 6–10 %, R2's 6 %, R4's decompression
  fallback), the finding is about the number, and the decision stands.
* the conduction face rule itself (harmonic mean, antisymmetric flux) — item 6
  says it is verified and asks that it not be re-examined, and I did not. B2 is
  about the law's *absolute scale anchor*, which is a different object from the
  face rule.
