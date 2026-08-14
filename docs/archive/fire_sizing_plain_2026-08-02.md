# The fire sizing decision — plain-language edition (2026-08-02)

**What this is.** Everything you need to make the one decision the fire arc
is paused on, in plain words, self-contained — no other document required.
The technical evidence lives in `fire_sizing_package_2026-08-02.md` and its
24 measurement files; every number here comes from there. Your three margin
questions are answered in section 6.

**What you are deciding.** One thing, seen from three angles: **how much
power your fires run on.** The draw radius sets how much oxygen a fire can
gather; the oxygen's energy content converts that into heating power; and
the heating power sets the temperature a burning crate settles at — which is
what the player sees and what ignites the neighbours. Pick the power, and
everything else gets calibrated around it.

---

## 1. What we measured, in plain words

A burning tile gathers oxygen from the air around it. Under your new
extended-draw law, "around it" means:

- **Radius 1** — only the 4 cells touching its faces (the old behaviour,
  proven bit-for-bit identical to the previous engine).
- **Radius 2** (currently shipped) — up to **12 cells**: the 4 faces plus
  the 8 cells two steps away, reached through open air only, with nearer
  cells counting more.
- **Radius 3** — up to **24 cells**, three steps out.

(Your guesses were right: 4, ~12, ~24.)

We pinned a fire at a fixed intensity so nothing else interfered, let the
air flow reach its steady state, and measured how much oxygen actually
arrives per second — converted to heating power using the standard
combustion fact that burning with one kilogram of oxygen releases about 13
million joules, for any common fuel.

| where | radius 1 | radius 2 (shipped) | radius 3 |
|---|---|---|---|
| open field | ~5.4 kW | ~12.7 kW | ~21.5 kW |
| sealed 12×12 ship room | ~4.3 kW | ~11.3 kW | ~19.6 kW |

For scale: 5 kW is a fireplace, 12 kW is a strong campfire, 21 kW is a
small bonfire, and a furiously burning wooden crate in a laboratory is
100–250 kW.

Two clean side-findings: a sealed room supplies only slightly less than the
open field (the room's own air is plenty *while it lasts* — your smothering
mechanic is about the total running out, and that inventory is conserved to
the last unit, verified). And opening a vent **to vacuum** empties a room
almost instantly — venting to space is explosive decompression, not a
breeze; sustained airflow needs somewhere for air to come *from*, which on
a ship means reservoirs or ambient sections — the map-design matter you
already set aside.

## 2. What the power buys: the fire's settling temperature

A burning object settles at the temperature where heat produced equals heat
lost (radiated away plus carried off by air). Because radiation grows
steeply with temperature, each power level buys a definite temperature:

- **~12.7 kW (radius 2):** the crate settles around **890 Kelvin — deep
  fire-orange.**
- **~21.5 kW (radius 3):** around **1000 Kelvin — bright orange.**
- The old target we wrote before anyone paid for it — 1170+ Kelvin,
  white-orange — needs **60–97 kW**, which no radius delivers.

The redeeming fact: **real campfires and wood fires burn at 900–1100
Kelvin.** The old white-hot target was itself unrealistically hot; it was
chosen for glow back when radiation cost nothing. The renderer paints
890–1000 K as exactly the deep oranges real wood fire has. So the
"shortfall" may in truth be the physics handing you the correct colour.

This is also what section 1.3 of the technical package says, translated:
*"the measured supply cannot pay for the old temperature target; either
accept a lower (more realistic) temperature, make the oxygen more potent,
or add another supply channel."*

## 3. The costs

- **Radius 2 over radius 1:** about 2.3–2.5× the delivered power, for a
  modest graphics-card cost (the combustion program grows from 40 to 70
  registers per thread — it still runs at half occupancy with zero
  spillover; in practice, cheap). **This is the good deal on the curve.**
- **Radius 3 over radius 2:** only ~1.7× more power, and the program grows
  to 168 registers — the card can then keep only ~17% of its threads busy
  in that pass. Fixable with a rework that's scoped but not built. **This
  is the expensive deal**, exactly as you sensed.

## 4. The "more potent oxygen" option, priced honestly

Making each unit of oxygen yield more energy multiplies every number in the
table without touching the draw. The price, and it lands exactly on your
ships: a sealed room's fixed air then supports proportionally more total
burning, so **smothering weakens by the same factor** — a room that would
choke a fire in two minutes takes twice as long at potency 2, five times as
long at potency 5.

- Reaching the old white-hot target at radius 2 needs potency ≈ 5 —
  smothering five times weaker. Probably unacceptable.
- A **bounded topping** of 1.5–2× at radius 2 lands at 19–25 kW — the same
  power as radius 3, without the graphics cost, at the price of smothering
  being 1.5–2× slower. A genuine middle path.

## 5. The menu (pick one, or mix — then everything recalibrates around it)

- **A — radius 2, no potency.** ~12.7 kW; fires settle deep-orange ~890 K.
  Cheapest, fully honest, smothering at full strength. The most "real"
  option.
- **B — radius 2, potency 1.5–2.** ~19–25 kW; bright orange ~1000 K;
  smothering 1.5–2× slower; no graphics cost.
- **C — radius 3 (with the rework), no potency.** ~21.5 kW honest; bright
  orange; smothering full strength; costs the register rework.
- **D — radius 3 + potency 2.** ~40 kW; approaching the old white-hot look;
  smothering 2× slower; costs the rework too.

**My view, clearly labelled as opinion:** start with **A**. It is the
cheapest, the most honest, it keeps your smothering at full strength, and
890 K is what real burning wood looks like. Judge the *feel* at the tuning
session with your own eyes — and if the fires read as weak, potency is one
config value, flippable in a minute, with its cost known in advance. That
is also exactly the "2b, and we try without making O2 more potent" you
originally ruled.

## 6. Your three questions, answered

**"Is radius 2 even worth it over radius 1?"** By the numbers, yes: it
better than doubles the delivered power for a genuinely small cost (section
3), and it is the mechanism that makes fire size respond to enclosure and
crowding at all. Radius 3 is where the economics turn poor. Your instinct
ordered the options correctly.

**"I don't fully understand 1.3."** Section 2 above *is* 1.3 in plain
words: it compares what the fire can gather (12–21 kW) with what the old
temperature target spends (60–97 kW), concludes the old target is
unaffordable, and lists the three ways out — lower the target (section 2),
potency (section 4), or a new supply channel (none exists; the wind level
needs an engine feature first).

**"The intensity requiring I > 2097 to survive — can we do anything about
it?"** Good news twice over. First: the number is **0.2097**, not 2097 — a
fifth of the intensity scale, not two thousand times its maximum. Second,
and more important: **it is not a law — it is a symptom of untuned dials.**
That measurement ran on the *shipped* config values, which have been known
fire-dead for weeks (they predate this whole arc; the shipped growth/decay
pair cannot sustain fire at any seed — the furniture "requirement" of 3.36
is above the maximum possible intensity, which tells you it's a broken
tune, not a threshold to meet). The recalibration that follows your sizing
decision sets the seed and the survival floor *together*, properly, at
whatever power you choose. And your reason for wanting a low seed — "lots
of headroom for things to burn more violently" — is exactly what the new
growth law protects: the seed only needs to clear the survival floor;
the *ceiling* is a separate dial, and the space between them is your
violence headroom, now independently tunable. No permanent seed-raising
needed; the measurement tools already pin intensity directly when they
need to see something.

**Jargon used in the technical package, decoded once:** *"knee / T-gate
limited"* = the fire died because it could no longer keep its own tile
above the minimum burning temperature (as opposed to running out of oxygen
or fuel). *"H_bed"* = the share of combustion heat that goes into the
burning object itself (the rest warms the air). *"Occupancy"* = how many
of the graphics card's threads can run at once in a given program; lower
means slower.

## 7. What happens after your answer

Nothing is building now, per your instruction. When your thought-through
answer arrives: the chosen package gets locked into the design, the
radiation-books patch builds against it, then the full recalibration in the
written order (which also revives natural burns and makes the smothering
curves measurable at last), then wind, embers, materials, and your
tuning-and-play session. The wind level's engine prerequisite (per-region
boundary conditions) goes on the design queue either way.
