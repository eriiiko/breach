# Body temperature: heat damage with inertia

**Status: PROPOSED, 2026-11-17. Nothing is built from this page until Erik has
read it.** Direction agreed in chat the same day (Erik: "body temp if it's
doable is exactly what I want"; "let's balance together later"; the 60 °C
limit may move). Raised by rung 2c's finding on #215: one frag beside a marine
dealt 30 blast damage and 576 heat damage in one tick.

## 1. What happens today

The heat row (`exchange.apply_environmental_damage`) reads the hottest tile
under a unit, turns it into a felt temperature (`exchange.felt_temperature`,
THE one function), and bills damage the same tick:

    damage per second = 1 + 0.05 x (felt - limit)      limit = 60 °C

The scale for fire light is anchored on the fire-safety literature (config.toml
`[combat]`, the table beside `heat_flux_to_temp`) and reads well: at the edge
of a fire nothing, in the flame 39 hit points a second.

Two things are wrong with it:

- **No inertia.** The body has no temperature of its own, so one tick of heat
  is billed as if it had lasted. Nothing warms up and nothing cools down.
- **No ceiling.** The frag's heat deposit reads as a felt temperature of about
  276 000 °C for one tick (computed back from the 576 hit points: 77 heat units
  on the tile). The scale was derived for radiant flux from a fire, not for a
  deposit from an explosive.

## 2. The law proposed

One new number per unit, its **body temperature** (integer, synced, a row of
the unit schema; starts at the ambient 20 °C that the felt temperature reads
with no heat).

Each tick, in the heat row's slot:

    felt  = min(felt_temperature(unit), felt_ceiling)
    body += (felt - body) x tick / time_constant
    damage per second = 1 + 0.05 x (body - limit)    when body > limit, else 0

So the damage formula and its literature table stay as they are; they read the
body instead of the moment. In a steady fire the body reaches the felt
temperature and the damage is exactly today's. What changes is the way there
and the way back.

| Dial | Start value | Meaning |
|---|---|---|
| `heat_time_constant_s` (per environment profile) | 2.0 s | How fast the body follows the heat. A suit with insulation has a larger one: this is heat resistance. |
| `felt_ceiling` (`[combat]`) | 3 000 °C | Nothing in the game reads hotter than a flame's core. Stops a deposit from reading as a star. |
| `temperature_max` (exists) | 60 °C | The limit. Erik's to move. |

The zombie's four-fold heat weakness stays where it is, in the damage
pipeline.

## 3. What it gives (computed from the law; to be measured in the patch)

| Situation | Today | Proposed |
|---|---|---|
| One frag adjacent (flash of one tick) | 576 hit points | the body jumps to about 82 °C and cools in 2 s: about 4 hit points. The frag's harm is its blast (30). |
| Two frags at once | over 1 100 | about 144 °C: about 15 hit points |
| In the flame (felt 820 °C) | dead in 2.6 s | dead in about 4 s; still burning for a second after stepping out |
| 10 kW/m² (felt 180 °C) | dead in 14 s | dead in about 16 s |
| A run through a flame, half a second | 20 hit points | about 5 |
| Edge of a fire (felt 60 °C) | nothing | nothing |

Sustained heat is dangerous, a little heat is not: the shape of the poison
law, from the physics of a body warming up.

## 4. What it costs

About the size of the poison law (one patch, one day):

- one row in the unit schema, so the host unit and the device unit store both
  carry it (generated layouts);
- the heat row rewritten on the host and in the device damage kernel, in Q16
  integers, the fraction `tick / time_constant` quantized once per profile;
- the pinned matches and the golden re-baselined once, with this page as the
  written reason (heat damage changes in every match with fire);
- the tile inspector and the HUD read the body temperature.

Left alone: what a learned soldier sees (it keeps the felt temperature; a new
observation is a new spec, later), the fire and radiation physics, the poison
law, every blast number.

## 5. Not in this patch (listed, not built)

- **Balance.** With the flash gone a frag beside a marine does about 34 hit
  points. Whether that is right is for the balance pass (Erik: "let's balance
  together later"; his idea of a data-driven system that tells what is
  overpowered is noted on #215).
- **Cold.** The same law would bill a body under `temperature_min`. Not now.
- **Body heat as a source** (a warm body that larvae seek, a body that warms
  the air). The number exists after this patch; no law reads it yet.

## 6. Order

After the grenade bounce, before the duel run of rung 2, so the soldiers do
not learn fire and grenade values that change under them.

## Systems

**(a) Existing systems this design uses**

- Coupling table (`src/simulation/exchange.py`): the heat row stays ONE row;
  felt temperature stays ONE function.
- Unit schema (`src/simulation/unit_schema.py`): the body temperature is a
  schema row in the same patch.
- Weapons / damage pipeline: the damage stays a `DamagePacket` through
  `apply_packet`; credit as today.
- Loadouts / suits (`[env_profiles.*]`): the time constant is a profile row.
- Unit store, device tables, device unit library: the device form of the row
  and the dial.
- Fixed-point kits; field digest + GOLDEN_AGGREGATE (one deliberate
  re-baseline).

**(b) New systems**

- **Body temperature** (`Unit.body_temp_q`, written only by the heat row).
  Draft rule: *a unit's body temperature is written by the heat coupling row
  alone; anything that warms or cools a body changes the felt temperature,
  never the body.*
