# L3 critique — scope, regression, and systems reuse

**Document under critique**: `docs/thermal_model_v1_design_2026-09-19.md`
**Lens**: scope / regression / canonical-systems reuse (one lens only)
**Date**: 2026-09-19 · repo at `fire-12`, `400eaa2`

Out of remit by instruction: §2's rulings R1–R9 and the ACCEPTED GAPs (human
decisions); physics correctness; integer/CUDA determinism. §8's L1 and L2 rows
were not used — every finding below was reached from the code.

**Summary**: 3 BLOCKING · 16 REQUIRED · 11 NOTE. Full table in §8.

**The single most important finding**: every test deletion in this arc is
scheduled *after* the patch that invalidates it — and the largest block of them
(the `cool_shift` surface, 34 tests in two dedicated files plus a CUDA check
pair) is owned by **no patch at all**, while T4b's gate reads "full suite" (B3,
with B2 as the same defect in the material loader).

---

## 1. §7.1 — canonical systems the design must use

### B1 — BLOCKING. Nothing owns where the per-tile values come from, and the one route the design implies is closed.

T1 says the two planes are *"the same shape as `heat_atten_q`"*. In the tree,
`heat_atten_q` is a **per-material projection**:

- `src/simulation/gamemap.py:1441` — `self.heat_atten_q = np.ascontiguousarray(tbl.heat_atten_q16[m], dtype=np.int32)`
- re-stamped per tile only when the **material** changes: `gamemap.py:1675` (`on_tile_changed`)

So "the same shape as `heat_atten_q`" means *a material column*. But the
rulings it is meant to serve are not material properties:

- **R2**: *"a floor grate, a deck over another deck and an open sky are
  different"* — two `hull` tiles differ by what is **above and below** them.
  A material column cannot express that.
- **R3**: *"room temp in ship, 0 K outside"* — a **regional** property.

The only per-tile authoring route that exists today is a material id, and
CLAUDE.md records that the **material ID budget is FULL** (arc #60) — a fact
§7.1 itself repeats (*"the id budget is full, so v1 adds no material"*).
Verified: `config.toml` carries exactly ten `[materials.*]` rows (`air`,
`hull`, `wood`, `door`, `steel`, `glass`, `furniture`, `door_closed`,
`kindling`, `foliage`) and CLAUDE.md notes id 9 numerically equals the CSV
`SPACE_CODE`.

The remaining route is authored per-tile level data, which is owned by two
canonical systems **§7.1 names neither of**:

- **Level data layer** — `level_lib.py` (write) / `level_loader.py` (read);
  CLAUDE.md: *"One writer ever — every tool is a client; never hand-write
  level.toml"*.
- **Map editor pattern** + **Editor UI from registry** (`tools/map_editor.py`,
  `tools/entity_editor_ui.py`) — how a per-tile value gets authored at all.

This is the exact failure mode this lens exists to catch: with no named source,
T1's implementer will invent one (a new level key, a new CSV layer, a bespoke
loader) rather than going through `level_lib`/`level_loader`.

**What would close it**: one sentence in §7.1 naming the source of each plane —
material column, level data layer, or "uniform-only in v1, authoring deferred"
— and, if level data, naming `level_lib`/`level_loader` as the systems T1 must
use. Note that "uniform-only in v1" is defensible (R3 says *"starting at room
temperature everywhere"*) but then §7.2's draft rule *"Filled at level load
through the Q16 boundary module"* claims a load path that will not exist, and
§6 property 2 is testable only by writing the plane directly from a test.

### R1 — REQUIRED. `src/temperature_scale.py` is not named, and T3 is nothing but SI→game conversions.

CLAUDE.md: *"Temperature scale | `src/temperature_scale.py` | The single
T_game→Kelvin map for bake, render, readouts, tools."*

T3's entire deliverable is literature numbers converted into game units:
`thermal_mass` from real `ρc`, `κ`, `ignition_temp` (*"~300–350 °C for
cellulosics"*, §4 item 10), heat of combustion. R3/R4 are stated in Kelvin
(*"room temp in ship, 0 K outside"*). And the canonical ambient in Kelvin
already lives there — `physics_runner.py:535`:
`self.eos.T_AMB_K = float(_ts.eos_t_amb_k)`, with `config.toml:869-885`
recording that `[physics.eos]` no longer carries `t_amb_k` *precisely so there
is one frame*.

A T3 report that states its own Kelvin→game conversion without routing through
`temperature_scale` is how a second scale is born — and G12 (issue #12's own
"collapse to ONE temperature frame") exists because that already happened once.

### R2 — REQUIRED. The coupling table (`src/simulation/exchange.py`) is not named.

Ledger item 11 (*"Unit heat damage — the `rad_flux` → damage law"*) and §6
property 9 (*"A marine beside a fire still burns, and a zombie takes 4×"*) are
both a **coupling-table row**. Verified consumer: `exchange.py:301`
(`rad_flux = getattr(gmap, "rad_flux", None)`) and `:325-330` (`fv = int(rad_flux[ty, tx])`).
CLAUDE.md: *"Coupling table | `src/simulation/exchange.py` | A physics→unit
coupling is one row, not plumbing."* Design v3's P3b names it explicitly
(*"`exchange.py` dials re-derived"*); this document drops it while keeping the
work (T4a *"units absorb (the body share)"*).

### R3 — REQUIRED. "Starting a fire" is not named, and §6's benches are the fixture class that went vacuous.

CLAUDE.md carries a long, emphatic canonical row: *"A fire is started by
DELIVERING HEAT, never by writing `fire` alone… Any new igniter (payload row,
weapon, editor tool, **test fixture, bench**) seeds the tile AT OR ABOVE its own
`ignition_temp` as well as lighting it, or it does nothing at all. Incidents,
all 2026-09: … and **five test fixtures silently measuring nothing**."*

§6 properties 6 and 10 are bench scenes with fires (*"a hot interior room"*,
*"a wood wall beside a plateau fire"*), i.e. exactly that fixture class, and
property 10 is a **negative** assertion (*"does not auto-ignite within the bench
window"*) — the one shape that passes for free when the fire was never lit.
Design v3's Systems section named this row; §7.1 drops it.

### N1 — NOTE. Tile inspector (`renderer/hover_readout.py::pack_hover_readout`).

CLAUDE.md: *"THE per-tile debug readout (F6 hides) — tools/HUD read it, never
roll a parallel field probe."* Design v3 landed Φ / `a`,`d` / `f` / `E°⁻¹(Φ)`
rows there at P1. Two new **per-tile** planes plus a HUMAN-TEST at T4b is the
situation in which somebody writes a one-off probe. One line in §7.1 (or a T1
sub-task) prevents it.

### N2 — NOTE. GameMap and the residency sets.

CLAUDE.md: *"GameMap | the single field store"*. The new planes are GameMap
fields, and the precedent for exactly this pair of properties is in the file:
`heat_atten_q`/`dyn_heat_atten_q` joined the resident set at `gamemap.py:159`,
and `cool_shift` — the per-tile plane R1 deletes — is at `gamemap.py:202` with
a comment explaining why. Not naming GameMap is minor; not naming the residency
set is how a plane works on the per-call path and is stale on the resident one.

### N3 — NOTE. Ingress lint + float ratchet.

CLAUDE.md lists both as canonical gates. §7.1 names the digest, GOLDEN_AGGREGATE,
the A/B harness and the CUDA harness but not these two, although T1 adds two
number-doors. (Substance is L2's; flagged here only as an inventory gap.)

---

## 2. §7.2 — new systems: duplication and should-be-extensions

### R4 — REQUIRED. `k_leak_q` is not a new system; it is a member of the Extinction planes system.

The scalar **already goes through the extinction-planes boundary module**:

- `src/simulation/physics_runner.py:465` —
  `self.k_leak_q = int(_optics_fx.quantize_scalar(_k_leak))`
- `src/simulation/optics_fixed.py` docstring — *"THE BOUNDARY MODULE for every
  extinction coefficient the radiation sweep's channels see"*, and it RAISES
  outside [0,1] rather than clamping.
- `cpp/src/radiation_sweep.cpp:160-162` already validates
  `k_leak_q ∈ [0, FP_ONE]` beside the `0 ≤ a ≤ d ≤ ONE` ingress check
  (`:186-191`).

So the per-tile promotion inherits the extinction row's rules wholesale: the
door, the raise-don't-clamp rule, the ingress invariant, the digest question.
Declaring a **second** canonical system restates all of that in a row that can
drift from the first. CLAUDE.md's own rule: *"if it almost fits, extend it —
never build a parallel copy."*

Stated honestly against the design: `k_leak` is not an extinction coefficient in
the optical sense (it does not attenuate in-plane transport; it is an
out-of-plane branch), so "extend the Extinction planes row" is a judgement, not
a tautology. But it is the judgement the evidence supports, and either way the
design must **decide** rather than default to a new row.

### R5 — REQUIRED. The `t_amb_q` draft rule contradicts two standing canonical rules.

The sweep's `t_amb_q` is not a sweep-local dial. It is bound from the
engine-wide ambient:

- `physics_runner.py:1053` and `:1459` — `t_amb_q=self._eos_t_amb_raw()`
- `physics_runner.py:1859-1863` — `_eos_t_amb_raw()` is *"the SAME fold
  `EOSSolver::step` does"*, derived by `GameMap._gas_energy_t_amb_raw`
- the same quantity runs `combustion.cpp:174, :725, :755` (the born-at-ambient
  rule), `bulk_transport.cpp:351-366, :468` (*"MINTED at ambient"*), and the EOS
  pressure calibration (`temperature_scale.eos_t_amb_k`).

The draft rule as written — *"THE per-tile ambient reference for **every**
thermal boundary; the sweep's ambient stream and **every future boundary
condition** read it, **never a global scalar**"* — if written into CLAUDE.md
would contradict the **Gas energy seam** row (*"MINTED mass (no gas donor) is
born at ambient"*, on the global) and would make arc #54's born-at-ambient
identity per-tile. That is either an engine-wide change T1 does not budget, or
the rule is overbroad. Scope it to the sweep in writing.

Second collision, smaller: `src/simulation/ambient.py` already exists and is
canonical in the survey (A29) — *"Ambient boundary derived constants… BOTH the
loader and `GameMap` must agree to the LSB, so the derivation lives here,
once."* It owns the ambient **gas** reservoir, not temperature, so this is a
near-miss rather than a duplicate — but a new system called "per-tile ambient"
that never mentions `ambient.py` is a naming collision that T6 will have to
untangle.

### R6 — REQUIRED. "The heat-count currency" is declared a system with no code home.

Every other row in §7.2 and every row in CLAUDE.md's inventory points at a file.
This one points at `report_p2b.md`. CLAUDE.md's rules lifecycle: *"At
implementation, draft rules are written into the project CLAUDE.md pointing at
the real code — CLAUDE.md only ever lists what exists."* As drafted, T6 is
being asked to write a canonical row whose "Where" column is a report.

The candidate home is the existing scale module (`src/temperature_scale.py`,
which already owns the one T_game↔Kelvin map and `eos_t_amb_k`) — i.e. this is
plausibly an **extension** of the Temperature-scale system (an energy scale
beside the temperature scale), not a new one. The design should say which, since
T2's deliverable is exactly "where does `c_v` live".

### N4 — NOTE. The two planes need two different doors, and §7.2 says "the Q16 boundary module" (singular).

`optics_fixed.quantize` **raises** outside [0,1] — right for `k_leak_q`, fatal
for a temperature. A temperature plane's door is `temperature_scale` /
`gas_fixed`. Small, but the survey already flags six byte-identical `*_fixed.py`
copies of one rounding rule (§E item 6); a seventh written by accident is the
predictable outcome of an ambiguous singular.

### N5 — NOTE (favourable, with one caution). The Dirichlet row is correctly scoped.

*"draft rule, NOT BUILT in v1"* with the EOS constraint stated is the right
shape. Caution for T6: CLAUDE.md must **not** receive this row in v1 — the
lifecycle rule is that CLAUDE.md lists only what exists. Keep it in the design
doc.

---

## 3. §7.3 — the `cool_shift` retirement surface

§7.3 lists: *"the ambient-decay dial and its vacuum variant, the
`COOL_SHIFT`/`COOL_SHIFT_VACUUM` globals, the per-row column, and the Pass-3
relax-to-ambient block"*, plus the thermostat ledger group.

`git grep -i cool_shift` over tracked non-doc files returns **41 files**. The
listed four items are a minority of it. What §7.3 misses, checked:

### B2 — BLOCKING. A load-time material validator is *denominated in* `cool_shift`, and nothing owns it.

`src/simulation/materials.py:796-805` runs a fire-seed sustain check at table
build time whose gain is, verbatim:

```
exp  = int(self.cool_shift[idx]) - int(self.heat_inv_shift[idx])
gain = bed_per_I * (2.0 ** exp)
i_sustain = (float(self.fire_T_ext[idx]) + span * h_min) / gain
if seed >= 1.15 * i_sustain: continue          # else warn
```

documented at `:706-740` (*"in == out gives the gain above"*): the check exists
because the tile's combustion deposit **is shed at `cool_shift`**. R1 deletes
that loss channel and replaces it with three channels (the sweep, `k_leak`,
conduction into gas), none of which is a per-tick shift this load-time formula
can evaluate. So the check is either silently deleted or silently wrong — and
what it guards is *"will a seeded fire stay lit"*, which is precisely what
T4b's HUMAN-TEST asks Erik to judge by eye.

This is BLOCKING because it is not a cleanup item: it is a derived calibration
that has to be re-derived or retired **with a decision**, and no row of §5 owns
it.

### B3 — BLOCKING. Two whole dedicated test files (34 tests) and a CUDA check pair die with `cool_shift`, and no patch owns them — while T4b's gate says "full suite".

| file | tests | why it dies |
|---|---|---|
| `tests/test_cool_shift_axis.py` | **25** | its subject *is* the per-material `cool_shift` column (docstring: *"THE FIX (this patch). A per-material `cool_shift` column"*) |
| `tests/test_temperature_cooling.py` | **9** | its subject is the Pass-3 ambient-cooling pass and the `COOL_SHIFT`/`COOL_SHIFT_VACUUM` pair |
| `tests/test_cuda_cool_shift.py` + `tests/cuda_cool_shift_check.py` | 1 + a check script (43 refs) | the CUDA Pass-3 twin |
| `tests/cool_shift_axis_gate_a_capture.py` | — | a committed gate-A capture artifact for the axis |

Partial hits in a further 14 test files (`test_thermal_mass_axis.py` 7,
`test_eos_p2_sealed_room_energy.py` 16, `test_fuel_fraction_axis.py` 5,
`test_fire_heat_source.py` 7, `test_temperature_convert.py` 3,
`test_fire_feedback.py` 3, `test_eos_p4_combustion.py` 4,
`test_temperature_conduction.py` 2, `test_radiation_sweep_gates.py` 2,
`test_radiation_sweep_shadow_wiring.py` 2, `test_pr3_capacity_law.py` 1,
`test_thermostat_books.py` 1, plus `_phase3a_driver.py`,
`fuel_fraction_axis_gate_a_capture.py`, `o2_full_reference_gate_a_capture.py`).

**None of these are among design v3 §11.1's 46 old-law tests** — that table
covers the *radiation* law (`test_pf1a_radiation_books.py`,
`test_pr1_fire_plane_cast.py`, `test_fire_heat_source.py`,
`test_heat_attenuation.py`). T5's row inherits §11.1 and §11.3 only. So the
cool-shift test surface is owned by no patch, and T4b — the patch that deletes
`cool_shift` — carries the gate **"full suite"**, which it cannot be.

### R7 — REQUIRED. A second live snapshot test blocks T4b, from the *other* half of the model swap.

`tests/test_thermal_mass_axis.py:76`
`test_existing_solid_materials_keep_their_tuned_thermal_mass` pins **every
material's `thermal_mass`** against an expected dict (`:87-94`, failure message
*"an EXISTING material's thermal_mass moved: expected …"*). R5 + T3 + T4b move
every one of them (steel 34→32, glass 17.8→16, and the rest). This test must
fail at T4b by construction.

It is also the case CLAUDE.md's 2026-09-08 rule describes (*"Tests assert
properties, never snapshots… a failing pre-existing test is a finding to
report, never a target to bend the design to"*) — so the right move is an
explicit disposition with rationale in the patch that moves the rows, not a
discovery during the gate run.

### R8 — REQUIRED. The rest of the surface §7.3 does not name.

Checked, file and line:

- **The per-tile plane itself.** `GameMap.cool_shift` is a real field
  (`gamemap.py:1489`), a member of the **resident synced set** (`:202`, with a
  comment on why), stamped on topology change (`:1697`), and exposed as
  `device_ptrs()["cool_shift"]` (`:190`). §7.3 names "the per-row column" but
  not the plane it projects to.
- **The validator and its bounds.** `materials.py:130` `_COOL_SHIFT_MAX = 20`,
  the `SHIFT_MIN` floor (`:461`), the `COOL_SHIFT` default in the thermal
  defaults (`:111`), and the vacuum-offset rule (`:434`).
- **The runner bindings.** `physics_runner.py:315-325` binds
  `cool_shift`, `cool_shift_vacuum` and `cool_shift_floor`, with a comment
  recording that the globals *"keep two live jobs"* (solver fallback **and** the
  vacuum offset) — deleting the column without the pair leaves the offset rule
  dangling.
- **The CUDA twin and a positional counter enum.** `cuda_temperature.cu:105`
  `C_COOL = 3` and `:115` `C_THERMOSTAT = 12`, written at `:467` and `:470`,
  folded **by index** at `physics_engine.cpp:380` and `:389`.
- **`e_cool_sum` and its seven consumers**: `temperature_solver.h:477`,
  `temperature_solver.cpp:682`, binding `bindings.cpp:1983`, and readers
  `tests/cuda_conduction_check.py:108`, `tests/cuda_cool_shift_check.py:84`,
  `tests/cuda_thermal_mass_check.py:202`,
  `tests/test_eos_p2_sealed_room_energy.py:74/154/282/323/346`,
  `tests/test_temperature_conduction.py:503`, `tools/storm_ledger.py:217`.
- **Five tools**, none in design v3 §11.3's edit list for this reason:
  `tools/fire_tune_loop.py` (**43** references — it dials `cool_shift`),
  `tools/fire_tuning_lab.py` (4 — the bench §7.1 says to extend),
  `tools/storm_probe.py` (3), `tools/eos_p5_bake.py` (1),
  `tools/eos_p5_out/index.html` (1).
- **Three canon chapters**: `docs/architecture/engine/04_atmosphere_and_pressure.md`,
  `engine/07_notes_from_claude.md`, `mechanics/03_combat_and_weapons.md`.
- **~31 config.toml references** beyond the two dial lines — three long derivation
  blocks (`:513-524`, `:1021-1042`, `:1489-1499`) whose arithmetic is stated *in*
  `cool_shift`, plus the per-row `cool_shift` entries.

### N6 — NOTE. §7.3's "not currently named in CLAUDE.md" is true of the string, not of the thing.

CLAUDE.md's **Energy closure identity** row names the same mechanism under its
other name — *"the two-way ambient thermostat (Pass 3 relax-to-ambient, ERIK'S
RULING: a deliberate modelling boundary, not a bug)"* — and cites
`docs/gas_energy_thermostat_ledger_2026-08-30.md` and `test_thermostat_books.py`.
§7.3's second bullet covers the ledger group, so nothing is missed; but the
sentence *"the sweep is broader than that one name"* understates in the other
direction: there **is** a CLAUDE.md row to amend, and R1 overturns a decision
recorded there as Erik's ruling. Worth saying so in §7.3 so T6 does not read it
as an unruled cleanup.

---

## 4. §5 — the patch cut: broken intermediates and ungateable states

### T1/T2/T3 really do move nothing live — verified

- `config.toml:597` — `k_leak = 0.0`, *"DORMANT at 0.0 until the reach question
  is ruled"*. Promoting a zero scalar to a zero plane moves nothing.
- The sweep is still shadow: it writes `rad_net_sweep` / `rad_flux_sweep` /
  `rad_amb_sweep` / `rad_fluence` (`gamemap.py:592-595`), and the live consumers
  read `gmap.rad_net` / `rad_flux` / `rad_amb` (`:525`, `:566`, `:546`;
  `exchange.py:301`). Nothing downstream sees a sweep plane until the flip.
- `t_amb_q` enters the sweep uniform from `_eos_t_amb_raw()`
  (`physics_runner.py:1053`, `:1459`), so a uniform plane reproduces it.
- T2 and T3 are reports. (Minor: T2's gate column reads *"the #54 closure
  identity still closes"*, which implies something ran; T3's *"no code change"*
  is the honest form. Cosmetic.)

So the shadow-first shape works. Three things about the flip do not.

### R9 — REQUIRED. §5's own ordering constraint is not satisfied by §5's own table.

§5 states three constraints. The table satisfies one.

1. *"`cool_shift` cannot die before the sweep's radiation is live"* — **satisfied**
   (sweep live at T4a, `cool_shift` dies at T4b).
2. *"`k_leak` must be live before the flip (else interior rooms are adiabatic)"*
   — **not satisfied**. T1 makes `k_leak` a *plane*; it stays at its dormant
   `0.0`. It goes live at **T4b**, and the flip is **T4a**. By the design's own
   argument, the T4a→T4b window is the state it says must not exist.
   (In practice `cool_shift` is still alive at T4a, so solids are not adiabatic
   there — they are **double-cooled**, which is the condition R1 gives as its
   reason for deleting `cool_shift`. Either way the stated invariant is
   violated; the prose and the table disagree about which patch "the flip" is.)
3. *"exactly one moment where the game changes"* — **not satisfied as written**.
   T4a changes the game (the fold reads different planes, the Pass-1 clamp goes
   live, units start absorbing) and T4b changes it again. The claim is true of
   `main` only, because T4a is *"built on a branch, not merged"*. Say that.

### R10 — REQUIRED. The flip does not say which radiation plane survives, and for two patches there are two writers.

Verified order: `cast_fire_heat` runs at `physics_runner.py:874` (per-call) and
`:1275` (resident), **before** `engine.step_tail` (`:986`, `:1424`); the sweep
is step 2b inside `step_tail`, and since P2a it **zeroes its own outputs** at its
start (`radiation_sweep.cpp:231`, `rad_flux[i] = 0;`).

T4a says only *"the fold reads the sweep's planes"*. Two readings, both live:

- **(a) the sweep writes the live planes.** Then it zeroes them *after*
  `cast_fire_heat` wrote them, so the old law is silently wiped every tick —
  still costing a full ray cast per tick for two patches, and every test or tool
  that asserts the cast has an effect through `Simulation.step` now fails.
- **(b) the fold is repointed to the shadow planes.** Then `exchange.py:301`,
  `tools/storm_ledger.py:291`, and `pack_hover_readout` keep reading
  `gmap.rad_flux` **by name** — i.e. the old law — so T4a's own stated purpose
  (*"units absorb … a flip without it leaves marines un-burnable"*) is not
  achieved unless those are repointed too, which the row does not say.

Design v3 §11's preamble is explicit that this is why P3 was one patch: *"then
**one** flip patch in which the old heat law, its tests and its tools die
together and units start absorbing (`rad_flux` must not lose its writer, row
24)"*. Splitting it into T4a/T4b/T5 is a legitimate re-cut, but it must then say
which writer owns which plane in the window it creates.

### R11 — REQUIRED. "A/B lockstep" cannot be T4a's gate in its documented meaning.

`tests/field_ab_harness.py` docstring: *"every sim field, every cell, every tick,
the refactored path must match the old path"*; CLAUDE.md: *"The refactor gate —
never prove a refactor with whole-grid means."* T4a deliberately changes
behaviour, so there is no pair of paths that must agree. Listing it as a gate is
vacuous (T4a against itself) unless the row says **what two paths** it compares
— plausibly CPU-vs-CUDA, which is what the neighbouring *"CUDA tol-0"* entry
already covers. Note this gate was inherited verbatim from design v3's P3, where
it had the same problem; the re-cut is the moment to fix it.

### N7 — NOTE. T4a is feel-adjacent and carries no human gate.

Design v3 §11.5 made **P3b** (*"units absorb: per-unit `heat_atten`, the body
share live, `exchange.py`'s dials re-derived"*) a **HUMAN-TEST** patch in its own
right. T4a absorbs P3a *and* P3b and its Human column is `—`. Because T4a is not
merged, CLAUDE.md's *"Feel-adjacent changes never auto-merge"* is not breached —
T4b's HUMAN-TEST covers the branch. Worth one line saying so explicitly, since
the table otherwise reads as a dropped gate.

### N8 — NOTE. The digest bump and the "one re-baseline" are in different patches.

T4a carries `dyn_heat_atten_q` into the digest at spec v6;
`tests/field_digest.py:71` has `DIGEST_SPEC_VERSION = 5` and `:121` salts the
version string into the hash before any field, and CLAUDE.md requires
*"membership/dtype change = version bump + regenerate all goldens same commit"*.
T4b is where the doc puts *"the one golden re-baseline of the arc"*. So as cut,
T4a must move goldens too. (The determinism lens owns the mechanism; recorded
here because it breaks §5's "one re-baseline" scoping claim.)

---

## 5. Silent contradictions of design v3 §11 and §12

### R12 — REQUIRED. The header claims to rule design v3 §12 item 5; §4 says it is still open.

Header: *"This document supersedes design v3 §12 items 5–8 and 13, **which it
rules**"*. Design v3 §12 item 5 is *"The currency pin value: 0.7 or 0.9
MJ/m³/K?"*. §4 of this document says: *"The only ruling left is the
**0.7-vs-0.9 pin** (item 3), which sets every row at once."* Items 6, 7, 8 and
13 are genuinely ruled (R1, R6, R2, R9). Item 5 is not.

This is not cosmetic: the pin is an **unscheduled input to T3**. T3's row and
gate say nothing about it, yet §4 states it *"sets every row at once"*, so T3
cannot produce final numbers without it. Either rule it before T3 spawns, or
give T3 a row that says "produce both columns, Erik picks".

### R13 — REQUIRED. §5 supersedes design v3 §11's P3 rows without saying so.

§5 replaces P3/P3a/P3b/P3c with T4a/T4b/T5 — different boundaries, different
ordering (the deletion moves *after* the human test), different agent tiers
(§11.5's P3c is **Sonnet**; T5 is **Haiku 4.5**, a tier Erik's 2026-09-15 rule
does not mention). But §5's closing line says *"Ray-engine-v2's remaining
patches (P4 … P7) follow **unchanged from design v3 §11**"*, which reads as "the
rest of §11 stands", and the header's supersession list covers only §12 and §9.
Design v3 is the design of record for the sweep; the P3 rows are now dead and
the document should say so in the header alongside §12 and §9.

### R14 — REQUIRED. `RAD_FLUX_CEILING`'s home is scheduled for deletion, and R9 keeps it "where it is".

Verified: `cpp/src/raycaster.h:352` — `static constexpr int64_t
RAD_FLUX_CEILING = (int64_t)INT32_MAX;`, used by `cuda_raycaster.cu:89`, cited
from `gamemap.py:561`, and tripwired by
`tests/test_radiation_sweep_shadow_wiring.py:444-460`.

Design v3 §11.4 archives `raycaster.{h,cpp}` at **P6** (`git rm`), and P4 deletes
`cuda_raycaster.{cu,h}`. Design v3 §12 item 13's "keep it" option was explicitly
*"(and re-home it as a named `[combat]` dial with a rationale)"*. R9 keeps it
without the re-homing; no row of §5 owns the move; §7.3 does not list it. Also
worth noting the new writer applies no ceiling at all
(`radiation_sweep.cpp:290`: `rad_flux[i] += abs_body - emit_body`), which is
consistent with R9's *"stops binding on its own"* but means the constant is live
only in code scheduled for `git rm`, and its tripwire dies with it.

*(Not a re-litigation of R9 — the ruling is "keep the ceiling". The finding is
that its file, its consumer and its tripwire all have scheduled death dates and
no patch owns the move.)*

### R15 — REQUIRED. Design v3 §12 item 9 is unruled, load-bearing for T3/T4b, and never mentioned.

*"`rad_scale_derived` is a global literal, but `A_rad` is per level … it assumes
`tile_size_m = 0.333` and a 2.5 m deck; `levels/airlock_demo` is at 1.0 m and
`levels/bench_two_room` at 0.5, where the emission is off by `tile_size/0.333`
(3× on the former). The precedent … is the opposite of a literal: the water
solver takes its `dx` from the level, never assumed."*

This is not in the header's supersession list and appears nowhere in the thermal
document — yet T3 derives every material constant against that same geometry
(`ρc` per m³, `κ` per m, `k_leak` from a 2.5 m deck), and §6's quantitative
properties (5, 6, 8, 10) are bench measurements whose answers scale with it.
Deriving a whole material table against an assumed tile size, one patch before
the arc's single re-baseline and its HUMAN-TEST, without saying which levels the
numbers are true on, is a scope risk the document should at least record.

### N9 — NOTE. Design v3 §12 item 11 deferred the `furniture` row to issue #68; T3 settles it now.

Item 11: the furniture row's `heat_atten = 0.5` / `light_atten = 0.55` /
`permeability = 0.5` (a half-empty tile) beside `thermal_mass = 8` (solid wood),
*"so the two should be settled together — **in the two-node session, issue
#68**"*. R6 defers #68 but T3 applies *"tailored effective numbers"* to furniture
immediately, which resolves item 11 by a different route. Defensible; the header
should add item 11 to what it supersedes so a later reader does not go looking
for the #68 session.

---

## 6. Test dispositions and §6's verification plan

### R16 — REQUIRED. T5 inherits §11.1 as "delete 46"; §11.1 is delete ~39, rewrite 6, keep 1 — and the rewrites gate T4.

Design v3 §11.1, read row by row:

| file | v3 disposition |
|---|---|
| `test_pf1a_radiation_books.py` (11) | delete |
| `test_pr1_fire_plane_cast.py` (11) | delete |
| `test_fire_heat_source.py` (14) | **delete 10, rewrite 4** |
| `test_heat_attenuation.py` (10) | **delete 8, rewrite 2** |
| `test_temperature_convert.py::test_air_tile_receives_radiation_deposit` (1) | **survives** |

T5's row reads *"the 46 old-law tests with rationale"* under tier **Haiku 4.5 /
mechanical**. Six of those are rewrites into **new end-to-end properties of the
sweep** (ignition across a gap at the derived reach; a marine burns and a zombie
takes 4×; bit-identity; the sweep is step 2b) — not mechanical, and not
deletable work.

Worse, they are **on the wrong side of the flip**. §6 property 9 (*"A marine
beside a fire still burns, and a zombie takes 4×"*, assigned to **T4a**) is
literally the rewrite of
`test_fire_heat_source.py::test_unit_next_to_fire_loses_hp_and_zombie_takes_4x`
(`tests/test_fire_heat_source.py:461`), whose file T5 deletes two patches later.
So the property is asserted at T4a by a test living in a file scheduled for
deletion, unless the rewrite moves to T4a.

### B3 (continued) — the "full suite" gate at T4a and T4b is unachievable as cut.

Checked: `tests/test_fire_heat_source.py` drives **full `Simulation.step()`** in
at least nine tests — `:329-347`, `:370`, `:394-415`, `:433-446`, `:461-484`,
`:498-512`, `:519-536`, `:569-617`, `:638-665` — pinning the old law end to end
(reach, ignition distance, unit damage, "wired into `Simulation.step`"). These
cannot survive T4a's fold flip or T4b's model swap, and both rows carry the gate
**"full suite"** while the file is deleted at T5.

Add the cool-shift surface of §3 (B3: 34 tests in two dedicated files plus a CUDA
check pair, owned by no patch at all) and `test_thermal_mass_axis.py:76`'s
material snapshot (R7), and the picture is structural: **every test deletion in
this arc is scheduled after the patch that invalidates it.** Design v3 avoided
this by making P3 one patch. The re-cut needs each deletion/rewrite to land in
the patch that breaks the property, with the rationale written there.

### N10 — NOTE. §6's ten properties cover T1 and T4b well; T4a is thin.

T1 gets four properties (1–4), T4b gets four (6–8, 10), T2/T4b share one (5).
**T4a gets one** (property 9, the marine burn) — and T4a is the patch that flips
the fold, turns on the Pass-1 clamp on CPU *and* the CUDA temperature twin, adds
the body share, and bumps the digest spec. Nothing in §6 pins the clamp going
live (`rad_clamp_hits` non-zero where expected, and the maximum principle
holding), nothing pins the fold reading the sweep rather than the cast, and
nothing pins the old cast's output no longer reaching a consumer — which is
exactly the ambiguity R10 identifies. Properties are cheap; this is the patch
that most needs them.

### N11 — NOTE. Five tool edits: the count is right, the list is stale for `cool_shift`.

Design v3 §11.3's P3-dated edits are `storm_ledger.py`, `fire_tune_loop.py`,
`storm_probe.py`, `fire_timing_harness.py` (+ `fire_tuning_lab.py`, already done
at P2b) — so "five" is fair. But those rows are about the *radiation* dials
(`T_emit_gate`, `cast_fire_heat` by name). The **`cool_shift`** edits to the same
tools are additional and unlisted: `tools/fire_tune_loop.py` carries **43**
`cool_shift` references (it dials it), `storm_probe.py` 3,
`fire_tuning_lab.py` 4, `storm_ledger.py:217` reads `e_cool_sum`.

---

## 7. What I checked and found sound

Said plainly, so the set knows what was covered and cleared:

- **The shadow-first shape works.** T1/T2/T3 genuinely move nothing live, for
  the reasons verified in §4 (dormant `k_leak = 0.0`; the sweep's four outputs
  are separate shadow planes no live consumer reads; a uniform `t_amb_q` plane
  reproduces the scalar binding). The design's central claim — everything
  provable lands before anything moves — holds for the first three patches.
- **§7.1's list, as far as it goes, is right.** The integer reference (reference
  first at T1), the radiation sweep, the extinction planes, the temperature
  solver, the gas energy seam and closure identity, the material/gas tables with
  the full id budget, `CFG`, the `*_fixed.py` doors, the four gates, and
  `tools/fire_tuning_lab.py` are all genuinely the systems this work belongs to,
  and each is stated with the right rule attached. The gaps in §1 are omissions,
  not wrong entries.
- **§7.3's second bullet is correct and non-obvious.** The ambient thermostat's
  ledger group (`e_thermostat_sum`) really is a separate retirement from the
  `cool_shift` dial, and *"the identity must close with the term removed, not
  zeroed"* is the right statement of it — verified against
  `temperature_solver.h:530-538` (the two counters are the same site under two
  names) and `physics_engine.cpp:380/389`.
- **R9's arithmetic checks out.** 459 % of the ceiling ÷ 2419 ≈ 0.19 %, matching
  *"about 0.2 % of the ceiling"*. The ruling's factual premise is sound; only its
  *home* (R14) is unowned.
- **T6 is the right patch and it is honest about its own scope.** *"`cool_shift`
  is not in CLAUDE.md today — verified — so the sweep is broader than that one
  name"* is true of the literal string, and I confirmed it: no CLAUDE.md row
  names `cool_shift`. (See N6 for the row that names the same mechanism under
  its other name.)
- **`k_leak`'s existing plumbing is already canonical.** It goes through
  `optics_fixed` at load (`physics_runner.py:465`) and is range-checked in the
  sweep (`radiation_sweep.cpp:160`). The per-tile promotion is a small, safe
  change *mechanically*; R4 is about which CLAUDE.md row owns it, not about the
  code.
- **No parallel solver, plane store or transport path is proposed.** The design
  adds channels and coefficients to the one sweep and the one temperature
  solver, which is exactly what the canonical rules ask for. The duplication
  findings above (R4, R5, R6) are all at the *rules* layer — rows that would be
  written into CLAUDE.md as new systems when they are members or extensions of
  existing ones — not code that rebuilds something.

---

## 8. Findings summary

| # | sev | finding |
|---|---|---|
| **B1** | BLOCKING | §7.1: nothing owns the **source** of the two per-tile planes. "Same shape as `heat_atten_q`" means a material column, but R2/R3's own examples are not material properties, and the material id budget is full. The level data layer (`level_lib`/`level_loader`) and the editor are unnamed and unbudgeted — the parallel-implementation risk this lens exists to catch |
| **B2** | BLOCKING | §7.3: `materials.py:796-805`'s **load-time fire-seed sustain check is denominated in `cool_shift`** (`gain = bed_per_I · 2^(cool_shift − heat_inv_shift)`). R1 deletes the channel it is derived from; no patch owns re-deriving or retiring it, and what it guards is "will a seeded fire stay lit" — T4b's HUMAN-TEST question |
| **B3** | BLOCKING | §7.3/§5: the **`cool_shift` test surface is owned by no patch** — 34 tests in two dedicated files, a CUDA check pair, a gate-A capture, plus partial hits in 14 more files — while T4b's gate is "full suite". Same defect for the old-law tests: every deletion is scheduled *after* the patch that invalidates it |
| R1 | REQUIRED | §7.1 omits `src/temperature_scale.py`, the single T_game→Kelvin map, while T3 is nothing but SI→game conversions |
| R2 | REQUIRED | §7.1 omits the coupling table (`exchange.py`), which owns ledger item 11 and §6 property 9 |
| R3 | REQUIRED | §7.1 omits the "Starting a fire" rule; §6's benches are the exact fixture class that went vacuous in 2026-09 |
| R4 | REQUIRED | §7.2: `k_leak_q` is a member/extension of the **Extinction planes** system (already quantized by `optics_fixed`), not a new canonical system |
| R5 | REQUIRED | §7.2: the `t_amb_q` draft rule ("never a global scalar", "every thermal boundary") contradicts the Gas-energy-seam born-at-ambient rule and collides with `src/simulation/ambient.py`. Scope it to the sweep or resize T1 |
| R6 | REQUIRED | §7.2: "The heat-count currency" is a canonical row whose Where column is a report; name its code home (likely an extension of `temperature_scale.py`) |
| R7 | REQUIRED | `test_thermal_mass_axis.py:76` pins every material's `thermal_mass` and must fail at T4b; dispose of it with rationale in the patch that moves the rows |
| R8 | REQUIRED | §7.3 misses the rest of the surface: `GameMap.cool_shift` + residency membership + `on_tile_changed` stamp, the validator bounds, the runner's three bindings, the CUDA `C_COOL`/`C_THERMOSTAT` positional slots, `e_cool_sum`'s seven consumers, five tools, three canon chapters |
| R9 | REQUIRED | §5's own ordering constraints: "`k_leak` live before the flip" and "exactly one moment where the game changes" are both violated by §5's table |
| R10 | REQUIRED | The flip does not say which radiation plane survives; two writers coexist for two patches, and design v3 made P3 one patch precisely to avoid it |
| R11 | REQUIRED | "A/B lockstep" is a refactor gate and cannot gate a deliberate behaviour change; say what two paths T4a compares |
| R12 | REQUIRED | The header claims to rule design v3 §12 item 5 (the 0.7-vs-0.9 pin); §4 says it is the one ruling left — and it is an unscheduled input to T3 |
| R13 | REQUIRED | §5 silently supersedes design v3 §11's P3/P3a/P3b/P3c rows while the closing line implies §11 otherwise stands |
| R14 | REQUIRED | R9 keeps `RAD_FLUX_CEILING` "where it is" — `raycaster.h:352`, a file design v3 §11.4 `git rm`s at P6, with its tripwire test |
| R15 | REQUIRED | Design v3 §12 item 9 (`rad_scale_derived` global vs per-level `A_rad`, 3× on a 1.0 m level) is unruled, unmentioned, and load-bearing for T3/T4b and §6's benches |
| R16 | REQUIRED | T5 inherits §11.1 as "delete 46"; it is delete ~39 / rewrite 6 / keep 1, the rewrites are not mechanical, and one of them is §6's own T4a property |
| N1 | NOTE | Tile inspector (`pack_hover_readout`) unnamed, with two new per-tile planes and a HUMAN-TEST |
| N2 | NOTE | GameMap and the resident-synced set unnamed |
| N3 | NOTE | Ingress lint / float ratchet missing from §7.1's gate list |
| N4 | NOTE | "the Q16 boundary module" (singular) — the two planes need two different doors |
| N5 | NOTE | The Dirichlet draft rule must not reach CLAUDE.md in v1 (lifecycle: CLAUDE.md lists only what exists) |
| N6 | NOTE | §7.3's "not in CLAUDE.md" is true of the string; the Energy-closure row names the same mechanism and records it as Erik's ruling |
| N7 | NOTE | T4a is feel-adjacent with `Human: —`; say that T4b's HUMAN-TEST covers the branch |
| N8 | NOTE | Spec-v6 at T4a vs "the one re-baseline" at T4b |
| N9 | NOTE | Design v3 §12 item 11 (the furniture row) is resolved by T3, not by issue #68 — add it to the supersession list |
| N10 | NOTE | §6 gives T4a one property, and it is the patch that changes the most |
| N11 | NOTE | The five tool edits are the radiation ones; the `cool_shift` tool edits are additional (fire_tune_loop alone: 43 references) |

