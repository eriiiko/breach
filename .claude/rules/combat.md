---
paths:
  - "src/simulation/weapons.py"
  - "src/simulation/combat.py"
  - "src/simulation/attack_resolver.py"
  - "src/simulation/damage.py"
  - "src/simulation/status.py"
  - "src/simulation/exposure.py"
  - "src/simulation/poison.py"
  - "src/simulation/stress.py"
  - "src/simulation/clearing.py"
  - "src/simulation/simultaneous.py"
  - "src/simulation/events.py"
  - "src/simulation/loadouts.py"
  - "src/simulation/draws.py"
  - "src/simulation/rng_salts.py"
  - "src/simulation/philox32.py"
  - "cpp/src/philox32.h"
---

# Combat: weapons, damage, draws, credit

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Weapons
- A weapon is a data row; every hit is a DamagePacket; nothing writes the hp but `damage.apply_packet`, in integers from the packet door (`DamagePacket.q()`) through `damage.mitigate_q`; temporary truths are status rows.
- Every packet carries its credit (#123): `attacker_id` (-1 = the environment) and a cause from `events.Cause`; the packet door refuses any other string.
- A row's cone, melee damage, melee push and reload stall are CLASS values read only through the Unit traits accessors.
- A row's other columns belong to their systems: `loudness` (Noise), `mag_size` / `reload_seconds` (Magazines; 0 = untracked), `expose` (Exposure), `bullet_g` / `muzzle_mps` (Unit impulse), `melee_push_ns` / `melee_arc_half_deg` (Melee verb). The carbine is one round a pull at 12 a second (Erik, 2026-11-10).

## Match draws
- EVERY random number the sim draws is ONE keyed Philox-4x32-10 block through `simulation.draws`: key = the sim's seed, counter = `(tick, draw_index, salt, unit_or_site_id)`, mapped to the outcome in integers. `draw_index` is LOCAL to the site, so no draw depends on any other (`tests/test_match_draws.py`).
- A new draw site is a `Salt` member (`src/simulation/rng_salts.py`) and a keyed draw: never a literal salt (scanned), never a numpy stream (the helpers refuse a Generator). `sim.rng` is drawn from by nothing in `src/`; `spawn_match` replaces it with `RefusingRng`.
- `MatchDraws.drawn` is a diagnostic, never hashed. The named integer sites stay float-free (`tests/test_p3e_no_float_sites.py`; a `p3e-door` / `p3e-view` mark names a constant quantized once / a render or observation view).
- Philox is vendored (Salmon et al. SC'11), held by `tests/test_philox32.py` (KAT + Python == C++).

## Exposure
- A new debuff-under-fire is a column here, never a status row or a second multiplier.
- START-OF-SLOT: inside an attack slot every hit reads the slot-start level, and the raises (and a death's zeroing) land together after the resolve's last effect; outside a slot they apply at once.
- Zeroed at death and conversion; `k_expose = 0` is DORMANT; hashed only while non-zero. The zombie's `expose_resist` is the zombie STATE's (`[zombie]`), anyone else's its env profile's. Integer, no draw.

## Poison
- Two per-unit amounts (depot, blood) moved by ONE integer law (`poison.tick`) that conserves what it moves and charges `hp_per_unit` for every count LEAVING THE BLOOD, as ONE POISON packet per tick through `apply_packet`: a dose's total harm is H × dose whatever the two times, which set only WHEN (Erik's law, 2026-11-14).
- Every poisoning enters through `poison.dose` from a coupling row. A new source of poison is a row that calls it, never a status row, a packet of its own or a second law (`tests/test_poison.py`: nothing but `poison.py` writes the columns, nothing but `poison.tick` builds a POISON packet).
- The blood is flushed at or below 1/64 unit only once the depot is empty, so every level returns to exactly 0 and the unit hashes as never poisoned. A zombie takes no dose. Zeroed at death and conversion. The rates are read through the TOX hooks' accessors.
- ACCEPTED GAPS: the latest dose takes the credit for all the poison; the POISONED status kind stays registered and unapplied (obs spec v1's layout), so an agent is not shown its levels until the next obs spec.

## Clearing step
- `clearing.drain` is THE per-tick drain of a unit state by a fraction of itself (the poison law's step, lifted in #215): poison calls it at Q16, stress at the finer `clearing.RATE_SHIFT` (2^-24). A new state that drains by a fraction calls it, never a copy; its device twin `unit_lib::clearing_drain` changes with it. Poison's arithmetic through it is frozen bit for bit (`tests/test_stress.py`).
- A half-life becomes a rate only through `clearing.half_life_rate` (ln 2 an exact rational, no libm).

## Stress
- Fed ONLY through `stress.produce`, a named producer row (`stress.PRODUCERS`) with its weight; the one producer is damage actually lost, from `damage.apply_packet` (host and device). A new producer is a row and its own ruling, never a write of `stress_q` (`tests/test_stress.py` scans both). Lowered ONLY by its own tick (slot 9c3'', fused into slot 9c3's device kernel). A zombie and a corpse take none; cleared at death and conversion.
- NOTHING in the game reads the stress or the heart rate until Erik rules an effect (2026-11-17: "readout in first patch is good, we wait with effects"); `tests/test_stress.py` scans `src/`. An effect reads `stress.heart_rate_q` (the one Hill curve, an integer table built at load), never the raw level; the heart rate is derived, never stored.
- The level is in 2^-24 points (`stress.LEVEL_ONE`): the clearing step truncates, so a Q16 level would bias a long half-life by percents.

## Simultaneous attacks
- An attack slot decides against its START state and applies together (#169): phase A walks units and rounds, decides eligibility, geometry and every draw, and RECORDS each effect (`hit` / `status` / `face` / `world` / `push`); `resolve()` applies them in walk order. So no unit's attack depends on its id, and a tick with no same-tick interaction is bit-identical to a one-pass slot.
- `face` is a facing TARGET, never a facing write. A hit's push is recorded right after its hit and applied only if the unit is alive at its turn. Exposure raises land together after the last effect.
- A new attack path takes `effects=` and records: never a direct write in phase A. `apply_packet` kills once. `IMMEDIATE` is for callers owning no slot (tests, tools; OnePhaseWEGO's timeline still fires one-pass: legacy debt).

## Credited cause
- A new damage source adds a `Cause` row in the same patch (the values are the event strings, so consumers compare literals; vacuum, pressure, O2, water and fall have no producer yet, so no row).
- Direct producers name their actor. A payload executes with its `owner_id`, which credits its blast and CLAIMS its heat and gas discs for THIS tick in `TickCredit`; the heat and poison rows credit the claimant of the hurting tile (the heat row only when the `heat` buffer is the exposure; the poison row through its dose).
- Claims die with the tick: a fire a unit lit, a cloud it left, a hot room's radiation are the environment's (ACCEPTED GAP, never a per-tile provenance plane). Indirect credit never enters `recent_attackers`.
- The attacker is HASHED in the per-tick unit digest; the hit's travel direction (`dir_x_q` / `dir_y_q`, #185) is NOT: render-only. The reward books consume these events (`attacker_id` -1 uncredited).

## Loadouts / suits
- The door refuses an unknown weapon, profile, field or count: never a per-unit literal (#119). An `explosives` count without a `charge` row is refused; `melee` must name a melee-archetype row with no crit. The unit's mass is body + `kit_kg`.
