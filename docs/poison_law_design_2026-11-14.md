# The poison law

**Date:** 2026-11-14. **Issues:** #29 (the patch before P6), #63 (the larva
bite that will feed it). **Frame:** `docs/unit_design_2026-11-14.md` section 6.
**Status:** the design of patch PX. The law and its first numbers are Erik's
rulings of 2026-11-14; the parametrisation, the integer form's details, the
credit and where it lives are the lead's (Erik: "credit, yeah, I'll leave it
up to you"). No play test before the merge (Erik). Every number is a
PLACEHOLDER for tuning; the principle for them is his: "as consequential as
possible without dominating the game".

---

## 1. The law

A unit holds two amounts of poison:

- `x0`, the DEPOT: what has entered the body and is not yet in the blood (a
  bite site, the lungs);
- `x1`, the BLOOD: what is hurting it.

Poison moves depot -> blood -> out, and it hurts as it leaves the blood:

    dx0/dt = -x0 / T_onset + input
    dx1/dt = +x0 / T_onset - x1 / T_decay
    damage rate = H x (x1 / T_decay)

One dose rises, peaks and decays (the Bateman curve). Three numbers describe
it, and they are the three Erik set:

| dial | meaning | first value (Erik) |
|---|---|---|
| `onset_seconds` (`T_onset`) | how quickly a dose reaches the blood | 4 s |
| `decay_seconds` (`T_decay`) | how quickly the blood clears | 20 s |
| `hp_per_unit` (`H`) | the hit points ONE unit of poison takes in total | 10 |

The unit of poison is one larva bite (Erik). With these values a single bite
takes 10 HP in all, hurts most about 8 s after the bite (0.33 HP per second
at the peak) and has done 95 % of its harm after about 65 s.

**Why "hurts as it leaves".** Erik's first form was `damage = k_dmg x x1`.
Writing `k_dmg = H / T_decay` is the same law, parametrised so that the total
harm of a dose is `H x dose` whatever the two times are: the dose and `H` set
HOW MUCH, the times set WHEN (his own sentence: "the dose sets how much harm,
k0 and k1 set the timing"). In integers it makes the total exact: every count
that leaves the blood is charged once.

There is no ceiling: a unit cannot be "fully poisoned" (Erik: "perhaps
not"). More poison is more harm.

## 2. The integer form

All amounts are Q16 ints (one unit of poison = 65 536 counts). Per tick, for
each living unit that is not a zombie and holds any poison or takes a dose:

    x0 += dose                                   (section 4; saturating at X_MAX, counted)
    m0 = step(x0, a_onset);  x0 -= m0;  x1 += m0
    m1 = step(x1, a_decay);  x1 -= m1
    damage_q = mul_q16_round(m1, H_q)            (Q16 hit points)

    step(x, a) = 0                      if x == 0
               = x                      if x <= FLUSH_Q
               = max(1, (x * a) >> 16)  otherwise

- `m0` and `m1` are each computed ONCE and applied to both sides, so no
  poison is created or lost (Erik's rule).
- `a_onset = round(ONE / (T_onset x tps))`, `a_decay` likewise: each an
  exact rational rounded once at the door. No exponential is evaluated (a
  libm call at load could differ by one count between machines). At 24
  ticks per second the first values are 683 and 137, time constants of
  3.998 s and 19.93 s.
- **The flush.** Below `FLUSH_Q` (1/64 of a unit: 0.16 HP at the first
  values) what is left moves in one tick. Without it the last few hundred
  counts would leave one count per tick for 20 s, each with its own hit
  event. With it a level always returns to exactly 0, so the unit goes
  dormant and its digest row disappears.
- `damage_q` goes through `damage.apply_packet` as ONE `DamagePacket` of
  type POISON per tick with `amount_q` (no float), so the mitigation, the
  exposure law and the events are the pipeline's own. A tick with `m1 == 0`
  emits nothing.

## 3. Where it lives

- `src/simulation/poison.py`, in the shape of `exposure.py`: a refusing door
  `poison_settings` over `[damage.poison]` (construction-bound, quantized
  once), `dose(unit, amount_q, source_id)` THE one way poison enters a unit,
  `tick(units, tick, events)` the law, `clear(unit)`.
- **Dials, `[damage.poison]`:** `onset_seconds`, `decay_seconds`,
  `hp_per_unit`, `gas_units_per_second` (section 4). The door refuses a
  missing or mistyped key by name, a time below one tick, and values that
  break the int64 bound written beside the law.
- **Columns** (unit-schema rows, group S, each hashed only while non-zero):
  `poison_x0`, `poison_x1` (Q16), `poison_src` (the unit credited with the
  latest dose; -1 = the environment or none).
- **Zeroed** at death and at a conversion (`poison.clear`).
- **Per-unit rates:** three hooks in `unit_traits` (after U1), each with its
  accessor: `TOX_ONSET` and `TOX_DECAY` on the two per-tick fractions,
  `TOX_HARM` on `H`. All `ONE` until stats are drawn (#199); the compile step
  will set them from the sheet (vitality).
- **Where in the tick:** one call line in slot 9c3, where
  `exchange.apply_poison_dose` is called today: first the gas row doses
  (section 4), then `poison.tick`.
- **Device twin:** P6.2 (a per-unit kernel with no order dependence).

## 4. Inputs

Everything enters through `poison.dose`. A zombie takes no dose (it does not
breathe and is immune to bites, Erik); the door returns without writing.

**Poison gas (this patch).** The coupling row keeps its read: the largest
poison density under the unit's footprint, nothing below
`[exchange] poison_min_density`. Instead of a damage packet it now gives a
dose:

    dose per tick = gas_units_per_second x density / tps

`gas_units_per_second = 0.6` keeps the TOTAL harm of breathing gas exactly
what it is today (`[exchange] poison_dps = 6` HP per second at full density
= 0.6 units x 10 HP), so the patch changes the timing and nothing else.
`poison_dps` is deleted. What changes in play, by the law:

- the harm starts late: in full gas a marine has lost under 1 HP after 4 s
  (today 24) and about 6.5 HP after 10 s (today 60);
- it does not stop at the edge of the cloud: every breath taken is paid in
  full over the next minute or so. Ten seconds in full gas is 60 HP,
  whenever the unit leaves;
- the LETHAL EXPOSURE is unchanged (about 17 s in full gas for a 100 HP
  marine), but death arrives later: a marine that never leaves the cloud
  falls after about 41 s, where today it falls after 17. A doomed unit keeps
  fighting for a while. Whether that is wanted is what the bench is for;
- so the poison grenade, the 40 mm poison round and the Miasma Vent become
  weapons one flees early or pays for later.

**Larva bites (after P6, with the bite patch).** The swarm adds each bite's
dose to a per-tile bite-dose field; a coupling row beside the gas row gives
the sum under a living unit's footprint to `poison.dose` (the unit design,
section 7.4). One bite = 1 unit.

**An antidote (wanted by Erik, later).** A second way out of the depot and
the blood that charges no damage. The law's form already allows it; it is a
verb, an item and a dial of its own design.

## 5. Credit

`poison.dose` records the credited unit of the dose in `poison_src`: for gas,
this tick's claimant of the densest footprint tile (`damage.TickCredit`,
exactly today's rule: a jet's sprayer, a gas payload's owner; a cloud that
only lingers is the environment's). The latest dose wins. Every later damage
packet carries that id with `indirect = True`, as today's gas packets do, so
it reaches the reward books and never `recent_attackers`. The event's cause
stays `poison_gas` until the bite patch adds its own cause.

## 6. What moves, and the gates

- **Sole cause:** every pinned match and golden WITHOUT poison is EQUAL on
  main and on the branch, tick by tick (unit records, field digests): the
  new columns are absent at zero and the law is dormant. Any pin that holds
  poison gas is re-baselined ONCE, with this document as its rationale; the
  report lists each.
- **Properties** (`tests/test_poison.py`; each names what must break it):
  - conservation: over any dosing schedule, `sum(doses) == x0 + x1 +
    sum(m1)` in counts, every tick;
  - the total harm of a dose is `H x dose` to within half a hit-point count
    per damaging tick;
  - one dose gives rise, one peak, decay; the peak lies where the discrete
    law puts it (the closed form of the two fractions), about 8 s at the
    first values;
  - a level always returns to exactly 0, and then the unit hashes as a unit
    that was never poisoned;
  - a zombie in gas holds no poison and emits nothing; a unit that dies or
    turns is cleared;
  - gas: at steady exposure the damage per second tends to
    `H x gas_units_per_second x density`; the dose is linear in density;
    nothing below `poison_min_density`;
  - credit: the damage that arrives after a unit left the cloud is credited
    to the dose's claimant, indirectly;
  - nothing but `poison.dose` writes the columns, and nothing but
    `poison.tick` emits a POISON packet from them (a scan of `src/`).
- **The bench** `tools/bench_poison.py` (Erik's tuning instrument): for a
  dose, and for N seconds in gas at a density: the damage curve, the peak
  and its time, the hit points lost at 5 / 10 / 30 / 60 s, the time to kill
  a 100 HP marine; and a poison grenade thrown into a room in a real match
  scene, before and after. Its table goes in the report and on #29.
- A panel line for the selected unit while it holds poison (`ui.model`
  data: the blood level and what is still to come), so the tuning can be
  watched.
- Full suite; `cuda_green` if a Class-P file is in the diff.

## 7. Accepted gaps

- ACCEPTED GAP: the `POISONED` status kind stays registered and unapplied
  (observation spec v1's layout is built from the number of status kinds);
  it is retired with the next observation spec version.
- ACCEPTED GAP: an agent is not shown its own poison levels in observation
  spec v1; it sees its hit points fall. The next spec version shows them.
- ACCEPTED GAP: the latest dose takes the credit for all poison in the body.
- ACCEPTED GAP: the bots do not know that leaving a cloud late still kills;
  the bot tournament will show what it costs them.
- ACCEPTED GAP: one hit event per poisoned unit per tick, as the gas row
  emits today, now for the length of the tail; the patch measures it.

## 8. Systems

**(a) Uses:** Unit schema (three rows); Unit traits (three hooks and their
accessors; U1 must be on main); the Coupling table (the gas row keeps its
read and reduction); Weapons / `damage.apply_packet` (the only writer of hit
points); Credited cause and `TickCredit`; Exposure (the module's shape, and a
POISON packet raises it as any hit does); the Tick conductor (one call line);
Fixed-point kits (`unit_fixed.mul_q16_round`, `div_round`); UI split.

**(b) Creates -- Poison.** Draft rule: "Poison is two per-unit amounts,
depot and blood (`poison_x0`, `poison_x1`), moved by ONE integer law
(`poison.tick`) that conserves what it moves and charges `hp_per_unit` for
every count leaving the blood, through `apply_packet`. Every poisoning enters
through `poison.dose` from a coupling row; a new source of poison is a row
that calls it, never a status row, a packet of its own or a second law. The
three per-unit rates are hooks."

## 9. Recorded for the later infestation patch (Erik, 2026-11-14)

Not built here. Any corpse, a zombie's included, can rise again, any number
of times, each time a new infestation reaches the threshold ("each larva can
infest infinitely many times"). A corpse can be DESTROYED beyond rising by
damage of any kind, whether or not it was a zombie when it fell: a corpse has
an integrity of its own.
