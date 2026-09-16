# Two-node thermal solids — design session prep (2026-09-16)

> **Status**: prep for a design session, not a design. It collects what the
> ray-engine-v2 arc has measured, states what is *not* the problem (so it is not
> re-litigated), and lists what the session has to decide.
>
> **Depends on**: `ray_engine_v2_design_v3_2026-09-15.md` §9 and rows 41–43;
> `ray_engine_v2_scheme_study_2026-09-13/report_p2b.md` §5, §9, §13;
> `fire_mechanics_inventory_2026-08-31.md`.
>
> **Related issues**: #61 (wood can't sustain fire), #12 (fire & heat tuning),
> #23 (fire never destroys furniture).

---

## 1. Why this session exists

P2b replaced the fitted radiation dial with one derived from physics. The
calibration itself came out well — it reproduces the survey's independently
derived reach curve to a fraction of a tile. But making the emission physical
exposed that **the receiver model cannot spend that energy correctly**, and the
gap is not a tuning gap.

The engine gives every thermal solid **one temperature per tile**. A tile is
therefore a single lump of uniform temperature: all of it heats together, all of
it cools together, and ignition tests that one number.

Real solids do not work that way, and the difference is not small.

---

## 2. What is NOT the problem

Stated explicitly, because the arc spent a session circling it and one of these
was wrongly escalated into a scheme question by the orchestrator.

**Emission and absorption are correctly linked, and there is nothing to fix.**
The sim carries exactly **one thermal band**. `heat_atten` is that band's
absorptivity, and the sweep uses the *same integer* for absorbing and emitting:

```
abs_mat = (stream * a_i) >> 16      // absorbed
emitted = (src    * a_i) >> 16      // emitted
```

Kirchhoff's law holds by construction — not as an approximation, as an identity
of the code. In a one-band model there is no spectral-selectivity question to
ask: real materials whose absorptivity and emissivity differ do so *across
wavelengths*, and a single-band model simply chooses one number and accepts the
compromise. That is a fidelity limitation of one-band radiation, shared with
every grey-body engine, and it is not an inconsistency. **Do not reopen it.**

**There is no reflection channel, and that is currently harmless.** Whatever a
cell does not absorb is *transmitted* onward, so incoming = absorbed +
transmitted, a complete accounting. Reflection would matter only for a polished,
low-emissivity surface used as a heat shield; every opaque row in the table is
at `heat_atten = 1.0` (painted/oxidised ship steel, which really is near-black
in the thermal IR). Worth revisiting only if a reflective material is ever
authored.

**Conduction does not break at a material boundary.** The face rate between two
materials is built from the **harmonic mean** of their conductivities
(`materials.py::_build_conduction_tables`, `hm = 2·ka·kb/(ka+kb)`), which is the
correct series-resistance combination — two thermal resistances in series add,
so a wood/steel face conducts at roughly the wood rate. The flux itself
(`temperature_solver.h::face_energy_q`) is **antisymmetric**: one integer,
applied to both cells with opposite signs, through `min(cap_i, cap_j)`. Energy
is conserved exactly across any boundary, and conduction and radiation are
separate passes, each booked into the arc #54 ledger. Correct as built.

---

## 3. The physics the current model is missing

Heat entering a surface diffuses inward. In time `t` it penetrates roughly

    d = sqrt(alpha * t)

with `alpha` the thermal diffusivity — about `1.5e-7 m²/s` for wood. Over a
45-second ignition exposure that is **2.6 mm**. Everything deeper is still at
ambient and is not participating in the surface's approach to ignition.

That is why thin kindling lights and a log does not, why a thick door resists a
fire a thin one does not, and why the same material can both ignite quickly and
store heat for a long time. **One temperature per tile cannot express any of
this**, because it forces "what ignites" and "what stores heat" to be the same
quantity.

---

## 4. What the arc measured

All at the derived (physical) calibration; sources in `report_p2b.md`.

| measurement | value | consequence |
|---|---|---|
| Bulk capacity of a furniture tile, pinned | **194 kJ/K** (0.277 m³ of wood, ≈139 kg) | the lump that must heat |
| Surface layer that actually ignites (2.6 mm over four faces) | **6.06 kJ/K** | `C_bulk / 32` |
| Heating rate 1 tile from a plateau fire | 39.0 kW → **0.201 K/s** | ignoring *all* losses |
| Time to ignition, lumped, no losses | **23 minutes** | real piloted ignition at that flux: **30–70 s** |
| Lumped model vs reality | **20–45× slow** | not a dial's worth of error |
| Steady state with the shipped `cool_shift = 13` | **67 game** at 1 tile | ignition temp is 280 — it never gets there |
| `cool_shift` needed for 1-tile ignition | **≥ 16** (8× today's e-fold) | a crate would then stay hot for 45 minutes |
| `thermal_mass` implied by the surface reading | **0.25** | the table cannot express it (power-of-two, and 0 means "gas") |

The last row is the crux: the two readings of `thermal_mass` — bulk inertia and
effective surface-layer capacity — differ by about **32×**, they want opposite
values, and **they are the same key today**.

---

## 5. Why this is very likely #61 as well

Issue #61: *"Wood can't sustain fire: conductivity 0.15 drains ignition heat with
a ~3 s e-fold."* That is the same defect from the other side. In a single-node
tile the ignition heat is immediately shared with the entire bulk and then
conducted away, so the tile cannot hold a surface hot enough to sustain
combustion. With a skin node the ignition heat lands on a small capacity, stays
hot, and drains into the core only through a finite conductance.

The current workaround is visible in the table: **`furniture` and `kindling`
carry `conductivity = 0.0`** — "no conduction face, like air" — purely so a
burning crate's heat stays on the crate. The `furniture` row's own comment admits
it is a fixture choice and warns against generalising from it. A skin node makes
that hack unnecessary and lets those rows carry real conductivity.

The session should treat #61 as a likely *consequence*, and check whether the
two-node model closes it rather than tuning it separately.

---

## 6. One row inconsistency found on the way

Independent of the two-node question, and worth settling in the same session
because it is the same confusion:

`furniture` carries `heat_atten = 0.5` **and** `thermal_mass = 8` — the identical
thermal mass as **solid wood**, whose `heat_atten` is `1.0`. The table header
defines `thermal_mass` as *"the analogue of volumetric heat capacity rho*c"*, and
the furniture row's own comment says *"wood-like (real rho*c ~0.7)"*, i.e. a full
tile of solid wood.

But the neighbouring columns describe a half-empty tile: `light_atten = 0.55`
(*"partial occlusion: a crate stack leaks some light"*) and `permeability = 0.5`
(*"smoke/air drift past crates"*). A tile that smoke drifts through and light
leaks past is not 139 kg of solid wood.

So either:

- `heat_atten = 0.5` is a **geometric fill fraction** (consistent with its
  neighbours) — and then `thermal_mass = 8` is roughly double what it should be;
  or
- `heat_atten = 0.5` is an **emissivity** — and then it is wrong for wood
  (ε ≈ 0.9) *and* inconsistent with the two columns that say the tile is
  half-empty.

Either way the row does not line up with itself. This also bears directly on
P2b's calibration, which pinned `V_tile` at the **full** solid column: if the
tile is partly empty, the same fill fraction should appear in both places.

---

## 7. What the session has to decide

1. **The node structure.** Skin + core is the obvious shape: a thin surface
   layer with small capacity that takes incident radiation and is what ignites,
   a core that holds the inertia, and one conductance between them. Confirm that
   two nodes are enough (versus a depth-resolved stack, which is almost
   certainly too expensive and too much state).
2. **Which node each channel touches.** Radiative deposit → skin. Neighbour
   conduction → from which node (skin-to-skin, core-to-core, or both)? The
   ambient thermostat → skin. Combustion heat release → skin. Gas-side
   convection → skin.
3. **What ignites, and what the igniter rule means.** `apply_temperature_ignition`
   reads which node? And the CLAUDE.md *"a fire is started by delivering heat"*
   rule — an igniter must seed *which* node at or above `ignition_temp`? Every
   existing igniter (payload rows, the flamethrower, the dev key, test fixtures)
   inherits that answer.
4. **The material columns.** Two new ones at least: skin capacity (or skin
   thickness, derived) and skin↔core conductance. Can skin thickness be *derived*
   from the existing `conductivity` and `thermal_mass` (it is `sqrt(alpha·t)` for
   a chosen reference exposure `t`), so the table grows by one column rather than
   two? Note the power-of-two constraint on `thermal_mass` and whether it must
   survive.
5. **The energy ledger.** Arc #54 P-G5's solid side is
   `solid_energy_books_sum = Σ thermal_mass_raw · T_raw`, a snapshot closing
   against four named channels. With two nodes it becomes two terms, and the
   skin↔core conduction is a **new booked channel**. The identity must still
   close in int64, exactly.
6. **Determinism and the digest.** A second temperature field is synced state:
   `DIGEST_SPEC_VERSION` bump, all goldens regenerated, `SIM_FIELDS`/residency
   sets, the Recorder's `DEFAULT_FIELDS`, and the CUDA twin.
7. **Render.** The blackbody glow and the tile inspector read which node? (Skin
   is the physically right answer — it is the surface you see.)
8. **Sequencing against ray-engine-v2.** P3a-2 (the flip) makes the derived
   calibration live, and on today's rows that means fire stops spreading. Does
   the two-node work land *before* the flip, or does the flip land with interim
   row values and the two-node work follow?

---

## 8. Blast radius

Everything the session must expect to touch:

- `cpp/src/temperature_solver.{h,cpp}` — all three passes
- `cpp/src/cuda_temperature.cu` — the twin, plus `TEMPERATURE_ENERGY_SLOTS`
- `src/simulation/materials.py` — new columns, the power-of-two constraint
- `src/simulation/gamemap.py` — the second field, residency sets
- `src/simulation/combat.py::apply_temperature_ignition`
- every igniter on the CLAUDE.md "Starting a fire" row
- `tests/field_digest.py` + spec → **v7**, all goldens regenerated
- `src/simulation/recorder.py` — `DEFAULT_FIELDS`
- `renderer/blackbody.py`, `renderer/hover_readout.py`
- the arc #54 closure tests (`test_thermostat_books.py`, the §6 benches)
- the project CLAUDE.md rows for the temperature solver and "Starting a fire"

---

## 9. Out of scope

- The emission/absorption model (§2) — settled, correct, closed.
- Conduction across material boundaries (§2) — correct as built.
- Reflection for heat — no consumer, no authored reflective material.
- Spectral/multi-band radiation — the sim has one thermal band on purpose.
- Trees and foliage rows — already out of scope for the arc (design row 6).
