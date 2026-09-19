# Thermal model v1 — design (2026-09-19)

> **Status**: design of record for the thermal half of the ray-engine-v2 arc
> (issue #12). Erik ruled the model on 2026-09-19; this document records those
> rulings *with their reasons*, derives what falls out of them, and cuts the
> patches.
>
> **Depends on**: `ray_engine_v2_design_v3_2026-09-15.md` (the sweep — §2.3 the
> scheme, §3 the planes, §9 calibration, §11 the patch plan, §12 the open
> questions); `report_p2b.md` (the derived calibration and its three findings);
> `report_p3a1.md` (the int64 widening and the flux-sensor ceiling);
> `thermal_solid_two_node_design_prep_2026-09-16.md` (issue #68, deferred).
>
> **This document supersedes** design v3 §12 items 5–8 and 13, which it rules,
> and design v3 §9's implicit claim that `cool_shift` survives the flip.

---

## 1. The one-paragraph statement

Radiation is now derived from physics rather than fitted (P2b), and the receiver
side has not caught up. This design makes the whole thermal chain honest in the
same currency: **energy arrives, energy raises temperature by `ΔT = E/C`, energy
leaves through channels that are computed rather than dialled.** The fake
loss channel (`cool_shift`) is deleted and its job returns to the two real ones —
radiation out of the deck plane, and conduction into air followed by advection.
Every material constant becomes a real number from literature. Nothing is tuned
to produce a desired behaviour; if flashover happens it will be because the
physics produced it.

---

## 2. Erik's rulings — settled, with their reasons

Each of these is a decision, not a finding. **Do not re-litigate them**; revisit
only at a patch boundary and only with Erik.

| # | Ruling | Reason |
|---|---|---|
| R1 | **`cool_shift` is deleted.** | It is a hand-rolled radiative loss — `temperature_solver.cpp:136` says so outright ("a wall radiating to space via `cool_shift_vacuum`"). Once the sweep is live, radiative loss is computed exactly and booked; keeping both counts it twice. Erik: *"with the new heat solver the drop to ambient is a boundary condition instead."* |
| R2 | **`k_leak` goes live and becomes per-tile.** | R1 makes the out-of-plane loss path load-bearing: with `k_leak = 0` a sealed interior room has no sky faces and is **radiatively adiabatic**. The floor and ceiling are real surfaces (~6 % of a cell's radiating area for a 2.5 m deck, the same ballpark as the design's derived ~0.10). Per-tile because a floor grate, a deck over another deck and an open sky are different. Erik: *"setting k_leak per tile depending on what's in it."* |
| R3 | **Ambient temperature becomes a per-tile plane**, starting at room temperature everywhere. | Erik wants *"room temp in ship, 0 K outside"* to be expressible. `t_amb_q` is a scalar today; making it a plane is the same change as R2 and they share one patch. |
| R4 | **Space is at room temperature**, not at 0 K, for v1. | *"most of the playing will take place inside the space ships."* A hot tile still loses to a room-temperature ambient, so the sink works; what is given up is the ship's net heat loss to space. |
| R5 | **`thermal_mass` stays a power of two**, with real `ρc` snapped to the nearest one. | It rides a bit-shift (`gain = deposit >> heat_inv_shift`), which is the solid path and is worth keeping. Measured cost on our actual materials: steel wants 34 and gets 32, glass wants 17.8 and gets 16 — **6–10 %**, well inside the model's other honesty. Erik: *"it's constrained for numerical reasons — and it's worth keeping."* |
| R6 | **No two-node (skin/core) thermal solid.** Walls take honest bulk capacities; **furniture takes tailored effective numbers.** | Erik: *"I don't want to increase the complexity in the model before we have a working simulation."* Consequence, accepted: thick structure heats 20–45× slower than real piloted ignition, so fire spreads through **contents** and largely not through **structure**. Erik: *"this is not a problem. They will still ignite from flamethrowers."* Deferred work is issue #68. |
| R7 | **Real numbers throughout. Nothing is tuned to produce flashover** or any other named behaviour. | Retuning happens after the engine works. This is the posture that would have prevented the 2419× fitted dial. |
| R8 | **Fixed-temperature (Dirichlet) tiles are a named extension point — SOLIDS ONLY.** | A fixed-temperature *wall* is a legitimate BC and is nearly free in this solver; it is how AC, radiators and heated compartments would be built. A fixed-temperature **gas** tile is forbidden: it would be an infinite energy source/sink inside the EOS and would destroy the gas simulation. Erik raised this himself and he is right. **Not built in v1.** |
| R9 | **The `rad_flux` ceiling stays where it is.** | It stops binding on its own: today's flux peaks at 459 % of `INT32_MAX` on the old cast's numbers, and the sweep's are ~2419× smaller — about 0.2 % of the ceiling. What remains is calibrating the **damage law**, which is a measurement (§4 item 11), not a cap ruling. |

### ACCEPTED GAPs

Deliberate omissions. These are decisions, not oversights; do not build machinery
to close them and do not report them as findings.

- `ACCEPTED GAP:` **A breach will not make a room cold.** R4 puts space at room
  temperature, so hull tiles never fall below ambient. Cold rooms, if wanted,
  come from the gas leaving (decompression cooling) or from level design.
- `ACCEPTED GAP:` **Floors and ceilings have no temperature of their own.** They
  are a radiative boundary (R2), not state. A tile is gas *or* solid and cannot
  be both, so giving the floor its own temperature needs a second field. Out of
  scope for v1.
- `ACCEPTED GAP:` **Floors cannot burn.** Same reason.
- `ACCEPTED GAP:` **Thick structure barely auto-ignites** (R6). Flamethrowers and
  direct heat deposits still light it.
- `ACCEPTED GAP:` **`thermal_mass` quantisation of 6–10 %** (R5).
- `ACCEPTED GAP:` **`c_p` is constant per material.** Real `c_p` for wood rises
  ~30 % between 293 K and 500 K. A linearisation, recorded.

---

## 3. What falls out of the rulings

### 3.1 The loss channels after `cool_shift`

A solid loses heat by exactly three routes, all computed:

1. **In-plane radiation** — the sweep. A cell hotter than its surroundings emits
   more than it absorbs, so `rad_net < 0`. Exact and booked.
2. **Out-of-plane radiation** — `k_leak`, to the floor and ceiling at the tile's
   ambient temperature, booked into `rad_amb`. **This is what replaces
   `cool_shift`.**
3. **Conduction into adjacent gas**, at air's real `κ`, after which the gas
   solver advects the energy away. Convection is therefore *emergent*, not a
   dial.

Note that `k_leak` is **not a time constant**. It is the fraction of each
ordinate's stream that leaves the deck plane per cell crossing
(`leaked = (i_in * k) >> 16`), with the ceiling returning ambient-temperature
radiation (`ret`). A cell at ambient is in exact balance; a hot cell loses net.
That is a decay toward ambient expressed as a conservative radiative exchange
rather than as a relaxation on temperature — which is precisely why it can be
booked and `cool_shift` could not.

### 3.2 The currency, and the gas side's hole

`cell_capacity_q` (`temperature_solver.h:220`) already produces **both** solid
and gas capacities in **one unit** — which is exactly why conduction can take
`min(cap_i, cap_j)` across a solid–gas face and mean something:

    solid:  capacity = thermal_mass
    gas:    capacity = N · c_v

So the *form* is unified. The *constant* is not: **`c_v = 1.0`**, and its own
config comment admits it — *"1.0 = neutral scale … no real-gas anchor yet."*

Working it through: at ambient (`N = 1.0`) an air tile behaves as
`thermal_mass = 1.0`. A `thermal_mass = 1` solid is `ρc = 87 500 J/(m³·K)`;
real air is `1.2 × 718 = 862 J/(m³·K)`.

> **Air is ~100× too heavy thermally.** A fire has to spend a hundred times the
> energy it should to raise the room's air temperature.

One complication for the patch to resolve rather than assume: arc #54's
`gas_energy = N · T_abs` has `c_v = 1` baked into its *representation*. So the
question is not merely the dial's value but **where `c_v` lives** — in the energy
field's definition, or at the conversion seam. §4 item 4 is that measurement.

---

## 4. The calibration ledger — every phenomenon, and what settles it

"Derivation" and "test" are both acceptable answers; a derivation *is* its own
test when it is written down and re-runnable. "Ruling" means Erik.

| # | Phenomenon | What it needs | Settled by |
|---|---|---|---|
| 1 | Radiation transport | the sweep's scheme | **Done** — gate 0 (bit-for-bit vs the integer reference) |
| 2 | Radiative emission scale | `rad_scale_derived` | **Done** — P2b derivation; reproduces survey §9.3 to a fraction of a tile |
| 3 | Solid heat capacity | `thermal_mass` per row, from real `ρc`, snapped (R5) | **Derivation** + the 0.7-vs-0.9 pin ruling |
| 4 | **Gas heat capacity** | `c_v`, and where it lives w.r.t. `gas_energy` | **Test** — §3.2; the sharpest open item |
| 5 | Energy → temperature | `ΔT = E/C` | **Derivation** — verified; it is the definition of heat capacity |
| 6 | Conduction solid↔solid | `κ` per row | **Derivation** from literature `κ`; **test**: a slab's e-fold against the analytic solution. The face rule (harmonic mean, antisymmetric flux) is already verified correct — do not re-examine |
| 7 | Conduction solid↔gas | the quantised face rate at air's `κ` | **Test** — and it is the channel R1 leans on |
| 8 | Out-of-plane loss | `k_leak` per tile | **Derivation** from deck geometry (~0.06–0.10); **test**: a hot tile's cooling curve |
| 9 | Emissivity | `heat_atten` per row | **Derivation** from literature; P2b proposed values, this arc applies them |
| 10 | Ignition | `ignition_temp` per row | **Literature** (~300–350 °C for cellulosics) |
| 11 | Unit heat damage | the `rad_flux` → damage law | **Test** — a marine's time-to-death beside a known fire, against a stated survivability target |
| 12 | Combustion | heat of combustion (J/kg), burn rate, O₂ demand | **Derivation** from literature; **test**: burn duration of a known fuel mass |

Items 1 and 2 are done. Items 3, 5, 6, 9, 10, 12 are derivations. Items 4, 7, 8,
11 want a measurement. The only ruling left is the **0.7-vs-0.9 pin** (item 3),
which sets every row at once.

---

## 5. Patch plan

Ordering constraint, and the reason the shape is what it is: **`cool_shift`
cannot die before the sweep's radiation is live** (nothing else would cool a
solid), and **`k_leak` must be live before the flip** (else interior rooms are
adiabatic). Meanwhile the flip on *today's* material rows produces a game where
fire does not spread. So everything provable lands first, in shadow, and there is
**exactly one moment where the game changes** — and it changes to the intended
model rather than to a broken intermediate.

| # | Patch | Mode | Tier | Oracle / gate | Human |
|---|---|---|---|---|---|
| **T1** | **`k_leak` and ambient become per-tile planes.** Two scalars become two Q16 planes (`k_leak_q`, `t_amb_q`), the same shape as `heat_atten_q`. **Reference first** (`sweep_ref_q.py`), then the engine. Still shadow — nothing live moves | subagent, worktree | **Sonnet 5** | **gate 0** bit-for-bit vs the reference; goldens unmoved; the uniform case must reproduce today's numbers exactly | — |
| **T2** | **The currency audit** (§3.2, ledger item 4). Determine what `c_v` must be for one gas joule to equal one solid joule, and **where it belongs** given `gas_energy = N·T_abs`. Deliverable is a report plus a *proposed* value — **applied at T4b, not here** | subagent, worktree | **Opus 5** | the #54 closure identity still closes; goldens unmoved (nothing applied) | — |
| **T3** | **Derive the real material table** (ledger items 3, 6, 9, 10, 12): `thermal_mass` from real `ρc` snapped to powers of two, `κ`, `heat_atten` as emissivity, `ignition_temp`, heat of combustion. **A report, not an edit** — the rows are Erik's, and applying them before the flip would move the live game twice | subagent, worktree | **Opus 5** | no code change; every number carries a literature citation | **review** |
| **T4a** | **The flip's mechanics.** The fold reads the sweep's planes; the clamp goes live on CPU **and** the CUDA temperature twin; units absorb (the body share — design row 24: a flip without it leaves marines un-burnable); `dyn_heat_atten_q` into the digest, spec v6. Built on a branch, **not merged** | subagent, worktree | **Opus 5** | full suite; the #54 closure identity; A/B lockstep; CUDA tol-0 | — |
| **T4b** | **The model swap**, same branch: `cool_shift` deleted, `k_leak` live at its derived value, the real material table applied, `c_v` corrected per T2, and **the one golden re-baseline of the arc** with written rationale | subagent, worktree | **Opus 5** | full suite; books close; one deliberate re-baseline | **HUMAN-TEST** — built, gated, pushed, **NOT merged**. Erik plays first |
| **T5** | **Delete the old heat law** — `cast_fire_heat` and both call sites, the dead dials, `RAD_LIM_SHIFT`, the pair budget, the CUDA cast bindings, the 46 old-law tests with rationale, the five tool edits (design v3 §11.1, §11.3) | subagent, worktree | **Haiku 4.5** | suite-gated; mechanical | — |
| **T6** | **CLAUDE.md rules walkthrough** (Erik's ask). Every canonical row re-read against the code after this arc removed systems; dead rows deleted, changed rows amended, new systems added per §6. `cool_shift` is *not* in CLAUDE.md today — verified — so the sweep is broader than that one name | inline | **Opus 5** | every row points at code that exists | **review** |

Ray-engine-v2's remaining patches (P4 CUDA twin, P5 gas/smoke, P6 light, P7
rules-side `light_q`) follow unchanged from design v3 §11 and are **out of scope
for this document**.

---

## 6. Verification plan — the properties to pin

Tests are written **from this list**, never derived from the implementation.
Each names the property and the change that must break it.

1. **The uniform case is unchanged.** With `k_leak` and ambient planes filled
   with one value, the sweep reproduces the scalar-era output integer for
   integer. *Breaks if* the plane indexing is wrong. (T1)
2. **A per-tile leak actually leaks per tile.** Two cells with different
   `k_leak_q` book different `rad_amb`. *Breaks if* the plane is read once and
   hoisted. (T1)
3. **Conservation survives the planes.** `Σ rad_net + Σ rad_flux + Σ rad_amb ≡ 0`
   in int64 with non-uniform leak and non-uniform ambient. *Breaks if* the
   ambient return is computed from the wrong cell's ambient. (T1)
4. **An ambient cell is still an exact fixed point** — per cell, with bodies, at
   its *own* ambient temperature. *Breaks if* `amb_m` is baked from a global.
   (T1)
5. **One gas joule is one solid joule.** A known energy moved across a solid–gas
   face raises the gas by the temperature its real heat capacity predicts.
   *Breaks if* `c_v` and `thermal_mass` are denominated differently. (T2/T4b)
6. **A sealed room is no longer adiabatic.** With `k_leak > 0`, a hot interior
   room's total energy falls; with `k_leak = 0` it does not. *Breaks if* the
   ceiling channel is not wired. (T4b)
7. **Nothing relaxes to ambient any more.** No solid's temperature changes
   except through a booked channel. *Breaks if* a `cool_shift` remnant survives.
   (T4b)
8. **The books still close over six groups** (#54 P-G5), with the ambient
   thermostat's term gone and `rad_amb`'s grown. *Breaks if* a channel is
   deleted without removing its counter. (T4b)
9. **A marine beside a fire still burns**, and a zombie takes 4×. *Breaks if*
   the body share is not live. (T4a)
10. **Thick structure resists and contents burn** — a wood wall beside a plateau
    fire does not auto-ignite within the bench window; a furniture tile does.
    This is R6's accepted consequence, asserted so it is deliberate rather than
    discovered. (T4b)

---

## 7. Systems

### 7.1 Existing canonical systems this design must use

From the project CLAUDE.md inventory. Using anything else in their place is the
bug.

- **Integer reference** (`sweep_ref_q.py` + its gates) — every arithmetic change
  is made **there first**, its gates re-run, and the C++ written against it.
  T1 in particular.
- **Radiation sweep** (`radiation_sweep.{h,cpp}`) — the one radiative transport
  path; new payloads are channels on it, never a second traversal.
- **Extinction planes** (`heat_atten_q`, `dyn_heat_atten_q`, `optics_fixed.py`) —
  the shape the two new planes copy; `optics_fixed` raises outside [0, 1], never
  clamps.
- **Temperature solver** — solids' temperature is derived here alone; gas
  temperature is the energy field's mirror.
- **Gas energy seam / energy closure identity** — every channel books itself;
  T4b deletes a channel and must delete its counter with it.
- **Material / gas tables** — a material is a table row; **the id budget is
  full** (arc #60), so v1 adds no material.
- **Config** (`CFG`, `config.toml`) — all tunables; solver params bound in
  `PhysicsRunner` only.
- **Q16 boundary modules** (`*_fixed.py`) — all Python↔field conversion; never
  hardcode 65536.
- **Field digest / GOLDEN_AGGREGATE / A/B lockstep harness / CUDA harness** —
  the gates. T4b carries the arc's single deliberate re-baseline.
- **`tools/fire_tuning_lab.py`** — the bench; extend it, never fork it.

### 7.2 New systems this design creates, with draft rules

Drafts live here; they are written into the project CLAUDE.md at implementation,
pointing at real code.

- **Per-tile ambient temperature plane** (`t_amb_q`) — *draft rule*: THE per-tile
  ambient reference for every thermal boundary; the sweep's ambient stream and
  every future boundary condition read it, never a global scalar. Filled at level
  load through the Q16 boundary module.
- **Per-tile out-of-plane leak plane** (`k_leak_q`) — *draft rule*: THE
  out-of-plane radiative loss path (floor and ceiling); a tile's leak is a
  property of what is above and below it. Never a temperature decay — it is a
  coefficient on the radiation stream, booked into `rad_amb`.
- **The heat-count currency** — *draft rule*: one heat count is a fixed number of
  joules, pinned once on the standard furniture row's real heat capacity
  (`report_p2b.md`); **every** thermal capacity in the engine, solid and gas, is
  denominated in it. A new thermal medium states its `ρc` in SI and converts
  here, never invents a scale.
- **Dirichlet (fixed-temperature) tiles** — *draft rule, NOT BUILT in v1*: a
  fixed-temperature boundary may be placed on a **solid** tile only. Never on a
  gas tile: that is an infinite energy source/sink inside the EOS.

### 7.3 Systems this design retires

Per the rules lifecycle, a retired system retires its rules.

- **`cool_shift` / `cool_shift_vacuum`** — the ambient-decay dial and its vacuum
  variant, the `COOL_SHIFT`/`COOL_SHIFT_VACUUM` globals, the per-row column, and
  the Pass-3 relax-to-ambient block. Its job moves to `k_leak`. **Not currently
  named in CLAUDE.md** (verified 2026-09-19), so T6's sweep is broader than this
  one name.
- **The ambient thermostat's ledger group** — #54 P-G5's `e_thermostat_sum` loses
  its channel. The identity must close with the term removed, not zeroed.

---

## 8. Adversarial critique — the lens table

Per the autonomous-patch-workflow: one lens per agent, **one at a time**, each
verdict written into this table **before the next critic spawns**. Lenses are
disjoint and no critic sees another's findings — independence is what the set
buys. Critics do not review the §2 rulings or the ACCEPTED GAPs: those are
Erik's decisions, not findings.

| lens | remit | verdict |
|---|---|---|
| **L1 — physics and thermodynamics** | Is the model sound? Are the three loss channels (§3.1) actually complete once `cool_shift` dies, or is there a fourth the design has not noticed? Is the currency argument (§3.2) right, and is the ~100× air figure correct? Does `ΔT = E/C` hold everywhere it is applied? Are the ledger's derivations (§4) the right ones? | *pending* |
| **L2 — determinism, integers, CUDA** | The two new planes against the number-ingress doors and `/fp:strict`; the digest and golden consequences; CPU↔GPU bit-identity for a per-cell leak and a per-cell ambient; the int64 headroom; the #54 closure identity with a channel deleted; whether T1's "uniform case reproduces the scalar era" is actually achievable bit for bit | *pending* |
| **L3 — scope, regression, systems reuse** | Does §7.1 miss an existing canonical system this design must use? Does §7.2 propose something that duplicates one? Does the patch cut (§5) leave a broken intermediate, or a state that cannot be gated? Are the test dispositions and the single re-baseline honest? Is anything in design v3 §11 silently contradicted? | *pending* |

**Synthesis** happens once, after the round, against this version of the
document. Blockers are resolved on paper — the document iterates (v2, v3…) until
it survives — and only then does T1 spawn.
