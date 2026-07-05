# The Physics↔Unit Exchange Layer

**Depends on:** [State & ownership](../engine/02_state_and_ownership.md),
[FieldEdit (write primitive)](../engine/13_field_edit.md),
[Determinism & the number-ingress rule](../engine/14_determinism_and_number_ingress.md),
[Units & entities](01_units_and_entities.md).
**Forward refs:** [Combat & weapons](03_combat_and_weapons.md) (damage types,
resistances, statuses get their full spec there).

Erik's founding principle, verbatim: **"there simply shouldn't be any barrier
between gameplay and the physics — if we come up with something we think is
interesting, we have to be able to implement it."** And its operational form:
**the unit must be able to read every field.**

This chapter systematizes that: ONE module, ONE table, ONE tick slot. Every
physics↔unit coupling — shockwave damage, heat damage, water slowing, gas
poisoning, O2, pushes — is a **row in a table**, not a plumbing project.
Adding a coupling is O(one row).

---

## 1. The read side: fields → units

Each coupling is a row:

```
(field, reduction over footprint, response(sample, unit.profile) -> outputs)
```

**Reduction vocabulary (v1):** `center | max | mean | sum | grad` — computed
over the unit's footprint tiles, all integer-exact on the Q16.16 fields
(`mean` = integer sum + one rounded divide; `grad` = footprint differences).
A response may produce **multiple outputs** (wave_p → damage *and* an impulse
push); outputs are quantized deltas/impulses/status applications.

**Tile resolution, not sub-tile (DECIDED 2026-07-04).** Units have Q16.16
sub-tile positions, but gameplay sampling is footprint-tile reductions —
fields are tile-resolution truth. Bilinear sub-tile interpolation (integer-
exact, 4 muls) is reserved as a *forces-only* upgrade if pushes ever feel
steppy. Not in v1.

**The coupling table (current known rows — this table GROWS; that is the
point):**

| Field | Reduction | Response → outputs | Status |
|---|---|---|---|
| `heat` | max | radiant flux → T_felt band → damage | ✅ shipped (`combat.apply_environmental_damage`) |
| `wave_p` | footprint sample | blast overpressure → damage | ✅ shipped (`apply_blast_damage`) |
| `wave_p` | grad | **impulse push**: `J = Σ_footprint(−∇p)·dt`, `Δv = J/mass` — the footprint sum scales with body area, so big-light units fly and small-heavy stand (density behavior from two existing stats); also the KNOCKED_DOWN trigger (ch. 06) | ✅ shipped (P4 — v1 uses `reduce_grad` (edge-line means, own-tiles-only: no wall-cell suction); the area-scaling Σ form returns as an explicit footprint-area factor when footprint sizes diversify) |
| `water_depth` | center | movement speed multiplier; **suffocation for non-water-breathers** | 📝 |
| `gas[teargas]` | max | density ≥ `teargas_blind_density` → `BLINDED` (can_aim off → snap-cone fire), refresh-stacked | ✅ shipped (weapons W3 — `exchange.apply_teargas_blind`, step 9c3; the sketched `mean` became **max**, the heat row's densest-tile-on-the-body reduction) |
| `gas[poison]` | max | density ≥ `poison_min_density` → one POISON packet/tick: `poison_dps × density / tps` through the mechanics/06 pipeline (zombie `resist_mult[POISON]=0` → immune, lazily skipped — no packet at all) | ✅ shipped (weapons W3 — `exchange.apply_poison_dose`, step 9c3 after teargas; the sketched dose-accumulation→status form simplified to direct per-tick packets, the heat row's idiom — a dose/status form can supersede it later) |
| `atmosphere` | mean | O2 partial → suffocation timer (skipped for units that don't breathe) | 📝 |
| `fire` | center | ignition → burning status | 📝 |
| `temperature` | max | contact/soak damage (distinct from radiant `heat`) | 📝 deferred (Erik #6) |

## 2. Per-unit response variation: the EnvironmentProfile

Different units respond to the same field differently — a fish breathes in
`water_depth` and suffocates in air; a robot ignores poison and O2; a marine
is the baseline. **The coupling rows are universal (the physics side); the
variation lives entirely in the unit's profile (the data side):**

- The response function's second argument is `unit.profile` — the species'
  `EnvironmentProfile` (already exists: `species.py`; combat already reads
  `temperature_max` / `environmental_damage_rate` from it), extended as rows
  land: `breathes = {air | water | none}`, per-hazard tolerances, and the
  **resistance vector** (per-damage-type multipliers/thresholds — RPG
  resistances: heat, blast, kinetic, energy, poison; full spec in the combat
  chapter).
- Profiles are **door-2 data**: config/species-table numbers, Q16.16-snapped
  at load. A new unit type with exotic responses = a new profile row, zero
  new code paths.
- **Unknown future couplings are pre-accommodated**: "we don't yet know all
  the things we might want" is fine *by construction* — a new hazard is a new
  coupling row + a new profile field with a sane default for every existing
  species.

## 3. The write side: units → fields

Exactly **two** write paths, both existing, no third ever:

1. **The occupancy stamp** (units block rays/heat — shipped, C++
   `stamp_units`): position → stamped tiles, before the ray pass.
2. **FieldEdit** for everything a unit *does* to the world: weapons,
   explosions, and future flavor (breathing CO2, footstep ripples, bleeding
   heat) — all deposits through the canonical write primitive
   ([engine/13](../engine/13_field_edit.md)), which quantizes everything that
   passes. This is enforcement-ladder L2 applied to unit writes.

## 4. The tick pipeline and THE ORDERING PRINCIPLES

Erik's question: *"should a laser act before a shockwave? does the actual
ordering matter, or only that it's a set order?"* The answer is both, split
cleanly:

**P0 — Fixedness is mandatory (determinism).** Whatever the order is, it is
the same on every machine: units in id order, couplings in table order,
statuses in list order, events in emission order. Ordering must never come
from dict/set/thread iteration or float ties. (This is the ingress rule's
sibling: an *order-ingress* rule.)

**P1 — Causality sets the macro-order: the tick pipeline.** Phases mirror
real causal flow; a phase reads what earlier phases produced *this* tick:

```
1. FIELD PHYSICS      — solvers advance all fields (GPU/CPU)
2. EXCHANGE READ      — the coupling table: fields → damage/forces/statuses
3. UNIT SIMULATION    — statuses tick; AI/orders; attacks resolve; movement
                        (consuming phase-2 forces & modifiers)
4. EXCHANGE WRITE     — stamps + FieldEdit queue flush (weapon/unit deposits)
5. CLEANUP            — end-of-tick clears (heat), event log seal
```

A weapon fired in phase 3 deposits in phase 4; its *field* consequences reach
other units next tick through phase 1→2. Nothing races, because reads and
writes live in different phases.

**P2 — Within a phase: simultaneity, by gather-then-apply.** All effects in a
phase are gathered against **start-of-phase state**, then applied as a batch.
Two lasers hitting one unit the same tick both hit a live target; a unit
killed this tick still completed its own already-committed phase-3 action
(mutual kills are allowed — physical, and WEGO-honest). And here integers pay
again: **batched Q16.16 HP deltas sum order-free** (integer addition
commutes exactly), so within-phase simultaneity is not approximate — it is
*bit-exact by construction*. Floats could not give us this.

**P3 — Travel time orders everything else, physically.** Instantaneous-vs-
propagating is a **weapon/field property, not an arbitration rule**: a laser
is hitscan (arrives in 0 ticks), a bullet arrives per its speed model
(instant or projectile — open weapon-design decision), a shockwave arrives
when the wave field reaches you (the solver's physical propagation speed).
"Does the laser act before the shockwave?" answers itself: whichever *arrives
at the unit* on an earlier tick acts first; same-tick arrivals are
simultaneous per P2. **No initiative tables** — the fields are the arbiter.
This is the no-barrier principle paying out: we never design turn-order
mini-games on top of physics that already knows the answer.

**P4 — Genuine tie rules are explicit data, not code accidents.** Where a
real semantic choice exists (death-timing edge cases, status stacking), the
rule is written in the combat chapter and config — never left to whatever the
loop happened to do. Current defaults: mutual kills allowed; heal+damage in
one tick net (order-free by P2's integer summation).

At 30–60 ticks/sec a tick is 16–33 ms — below human perception of order —
so within-tick simultaneity is free realism, and it is *exactly* the
semantics Erik's plan-both-phases WEGO control mode wants.

## 5. Statuses: DoT / HoT / buffs (requirement LOCKED, spec forward-ref'd)

Persistent afflictions are in: poison DoTs, regeneration HoTs, burning,
suffocation timers, buffs/debuffs, and the RPG resistance layer
(armor class, fire/heat resist, energy resist, armor-vs-kinetic).

**Where they live — an ownership clarification:** the *simulation* owns unit
health, and unit records (hp, statuses) are **CPU-side synced state** — the
GPU owns *fields* (per this chapter's division of labor; see §6). DoTs do not
need to be GPU-resident: tens of units × a few statuses is trivially cheap.
What they MUST be is **deterministic and digested**: integer tick durations,
Q16.16 magnitudes (door 2), fixed processing order (P0), applied through the
same quantized-delta pipeline as direct damage, and added to the unit-state
digest when they land (a legitimate golden re-baseline). Statuses tick at the
top of phase 3.

Shape sketch (full spec in the combat chapter): `StatusEffect(type,
magnitude_q16, remaining_ticks, source_id)`, per-unit list, per-type stacking
rule (`refresh | stack | max`), config-driven.

## 6. The performance contract (why this design is residency-ready)

The exchange layer is deliberately a **reduction interface**: units never
consume fields, only named reductions over footprints. Consequences:

- **Today (CPU mirrors / `--cuda` download-all):** ~64 units × ~8 couplings ×
  9 tiles ≈ a few thousand integer reads/tick — microseconds; never a
  bottleneck. Fields are 10⁴–10⁶ cells; units are 10¹–10² records — the five
  orders of magnitude that decide what belongs on which processor.
- **S8-residency world:** a trivial kernel computes the per-unit reductions
  on-GPU and downloads ONE buffer (~2 KB/tick) instead of whole fields
  (~MB/tick each). Upstream: stamps + the FieldEdit queue, also KB. The
  interface doesn't change — a second implementation swaps in behind it,
  gated bit-identical, exactly like the seven solvers were.
- **ML training:** the same on-GPU reductions ARE the observation extraction —
  policy inputs without host round-trips.
- **cudaunit swarms (forward idea):** GPU-resident simplified hordes plug in
  *behind this same interface* as another consumer; nothing here blocks it.

Rule of thumb this encodes: **GPU for fields (regular, massive, parallel);
CPU for agents (few, branchy, logical); a narrow reduction/deposit interface
between.** Interfaces are decided now; implementations swap at measured
forcing events (big maps, ML training, hordes) — never speculatively.

## 7. Determinism discipline (the ingress rule applied here)

All reads: integer fields → integer reductions. All responses: door-1/door-3
chains parameterized by door-2 profile constants. All outputs: quantized
(HP deltas via `quantize_hp_delta`; impulses in Q16.16). All orders: P0.
The exchange layer is the single most coupling-dense boundary in the game —
and by construction every number crossing it passes a door.

## 8. Open questions (tracked, owed to later items)

1. **Bullet speed model** — instant hitscan vs projectile-with-speed (weapon
   design, combat chapter). P3 absorbs either choice.
2. **Death-timing edge semantics** — P2 default (committed actions complete;
   mutual kills allowed) to be ratified in the combat chapter.
3. **Bilinear-for-forces** — only if push feel demands it.
4. **Observation-space reuse** — formalize exchange reductions as the ML
   observation schema (ml/ chapter, later).

---

## Implementation status (2026-07-04)

| Piece | Status |
|---|---|
| heat→damage row (max reduction, profile-parameterized band) | ✅ shipped |
| wave_p→blast damage row | ✅ shipped |
| Unit occupancy stamp (write path 1) | ✅ shipped (C++) |
| FieldEdit as the sole deposit path (write path 2) | ✅ shipped |
| EnvironmentProfile hook (temperature_max / env_rate) | ✅ embryo shipped |
| The formal coupling TABLE + reduction vocabulary | ✅ shipped (P1 refactor, 2026-07-05): `src/simulation/exchange.py` — `REDUCTIONS` (center/max/mean/sum/grad, integer-exact) + `COUPLING_TABLE` with the two shipped rows registered (responses moved verbatim; behaviour-preserving, digest unchanged). Rows still execute at their legacy tick positions |
| The named EXCHANGE-READ slot (rows consolidated, table-order execution) | 📝 later patch (P4-era) — note: today blast runs at *detonation sites* (grenade fuse-out, door explosives), heat at post-physics step 9c; consolidation must reconcile that split |
| wave_p→impulse push row | ✅ shipped (P4, 2026-07-05 — `exchange.apply_wave_push`, both outputs: the reduce_grad footprint read → `Δv = k_push·(−∇p)/mass` per tick (v1 stateless nudge, per-axis wall clamp + displacement cap) AND the KNOCKED_DOWN trigger (squares-compared vs `threshold × stability`). Runs at step 9c2, after the heat row (documented within-tick order: heat, then push). Calibration finding: a passing acoustic pulse's net impulse largely cancels (front push ≈ tail pull) — the visible motion is a 0.3–0.5-tile buffet; a sustained blast-wind throw would read the atmosphere dome (a possible LATER row, Erik's call). Standard values in `[exchange]`, config.toml — awaiting Erik's feel gate |
| gas[teargas]→BLINDED + gas[poison]→POISON DoT rows | ✅ shipped (weapons W3, 2026-07-05 — `exchange.apply_teargas_blind` / `apply_poison_dose` at step 9c3, after heat + push (the documented within-tick order: heat, push, teargas, poison). Footprint-**max** reduction, Q16.16 integer thresholds (`[exchange]` — Erik's standard-value dials), LAZY by construction: an all-zero plane costs one integer `.any()`; neither row takes a generator (no RNG ever), so every gas-free trajectory is bit-identical — the golden did not move. Poison's per-tick amount keeps the heat row's idiom (door-3 float on the door-1 read, quantized once at the HP boundary: full density = exactly 0.25 HP/tick at the standard 6 dps / 24 tps); immune units (`resist_mult[POISON] = 0` — zombies) draw NO packet (lazy emission, no 0-damage event spam) |
| water/O2/fire rows + breathes profile fields | 📝 designed, pend their systems (suffocation/burning triggers) |
| Statuses (DoT/HoT/buffs) + resistances | ✅ core shipped (resist tables P2; status system + the top-of-phase-3 tick slot P3, 2026-07-05 — `simulation/status.py`, digest `__unit_status__`). The coupling-row TRIGGERS (fire→burning, gas→poison, O2/water→suffocation, wave_p→knockdown) still 📝 with their rows |
| On-GPU reduction kernel | 📝 S8-residency era, behind the same interface |
