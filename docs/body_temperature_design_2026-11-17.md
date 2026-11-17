# Body temperature: heat damage with inertia

**Status: PROPOSED (version 2, the same evening), 2026-11-17. Nothing is built from this page until Erik has
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

## 2. The law proposed (version 2: in energy, no ceiling)

Erik, 2026-11-17, on version 1's ceiling of 3 000 °C: "We'll have plasma guns,
they must pay off"; "it must be heat energy, not just temperature. In vacuum
it gets really hot." Both right. The ceiling is gone and the law is an energy
balance. (The game's "felt temperature" already IS an energy flow in disguise:
`heat_flux_to_temp` was derived from kW/m² on the body. The law below reads
that flow directly.)

One new number per unit, its **body temperature** (integer, synced, a row of
the unit schema; starts at the ambient 20 °C).

Each tick, in the heat row's slot:

    energy in   = what the body absorbs this tick (radiation and deposits:
                  the same reading `felt_temperature` makes today, in joules)
    energy out  = loss x (body - ambient) x tick
    body       += (energy in - energy out) / heat_capacity
    damage per second = 1 + 0.05 x (body - limit)    when body > limit, else 0

In a steady fire the body settles where in equals out, and with the loss
number below that is exactly today's felt temperature, so the literature
table and today's steady damage are kept. What is new is the heat capacity:
the body takes time to warm and time to cool, and how far a burst of heat
moves it depends on how many joules the burst carries, not on how hot its
source reads.

| Dial | Start value | Meaning |
|---|---|---|
| `body_loss_w_per_k` (derived, not tuned) | 159 W/K | Fixed by the existing anchor: 2.5 kW/m² on the body holds it at the 60 °C limit. |
| `body_heat_capacity_j_per_k` (per environment profile) | 320 J/K | How many joules warm the shell by one degree. With the loss above it gives a time constant of 2 s. A suit with more insulation has more: this is heat resistance. |
| `temperature_max` (exists) | 60 °C | The limit. Erik's to move. |

The capacity is that of the suit's outer shell, not of the whole body: burns
are a matter of the surface.

A plasma gun pays off by delivering many joules; nothing clips it.

The zombie's four-fold heat weakness stays where it is, in the damage
pipeline.

**The frag.** Read as energy, the frag's heat deposit puts about 2.4 MJ into
a body on the next tile in one tick (computed back from the 576 hit points).
A real fragmentation grenade holds about 0.9 MJ in all, most of it in the
blast and the fragments. So the number that is wrong is the frag's heat
deposit, a placeholder, and the patch calibrates it: a body on the next tile
absorbs a few kJ, which warms the shell by 10 to 20 °C. Where that deposit
comes from (the payload row or the explosive's row) is the patch's first
finding to report.

**Stage 2, not in this patch: the air and the vacuum.** Today the heat row
reads radiation and deposits only; hot air by itself does not warm a unit.
The energy form gives the place for it: exchange with the air in proportion
to the air's density times the temperature gap. Thin hot gas then carries
almost nothing (a near-vacuum that reads thousands of degrees does not cook),
and in vacuum the same term stops cooling the body, so a heated suit stays
hot much longer. That is its own patch with its own numbers, once this one
is played.

## 3. What it gives (computed from the law; to be measured in the patch)

| Situation | Today | Proposed |
|---|---|---|
| One frag adjacent (flash of one tick, deposit calibrated) | 576 hit points | the shell jumps 10 to 20 °C and cools in 2 s: 0 to a few hit points. The frag's harm is its blast (30). |
| In the flame (50 kW/m²) | dead in 2.6 s | dead in about 4 s; still burning for a second after stepping out |
| 10 kW/m² | dead in 14 s | dead in about 16 s |
| A run through a flame, half a second | 20 hit points | about 5 |
| Edge of a fire (2.5 kW/m²) | nothing | nothing |
| A weapon that delivers 100 kJ in one hit (a plasma bolt, later) | n/a | the shell jumps about 300 °C: about 30 hit points over the next seconds |

Sustained heat is dangerous, a little heat is not: the shape of the poison
law, from the physics of a body warming up.

## 4. What it costs

About the size of the poison law (one patch, one day):

- one row in the unit schema, so the host unit and the device unit store both
  carry it (generated layouts);
- the heat row rewritten on the host and in the device damage kernel, in Q16
  integers, the capacity and the loss quantized once per profile;
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
- **The air and the vacuum** (section 2, stage 2).
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
