# Weapons & Combat Resolution

**Depends on:** [Units & Entities](01_units_and_entities.md),
[Physics↔unit exchange](05_physics_unit_exchange.md),
[Damage, health & conditions](06_damage_health_and_conditions.md),
[Ray Engine](../engine/08_ray_engine.md), [Smoke & gases](../engine/05_smoke.md),
[Temperature & fire](../engine/06_temperature_and_fire.md),
[FieldEdit](../engine/13_field_edit.md),
[Determinism & number ingress](../engine/14_determinism_and_number_ingress.md)

*Framework designed 2026-07-05 (agenda item 5, all core decisions blessed by Erik).
This chapter subsumes the earlier draft (rifle/grenade/door-charge + the energy-beam
design); everything still true from that draft is folded in below.*

A weapon in Breach is **a row of data, not a system**. The code knows a small,
closed set of *delivery archetypes* — the physically distinct ways an attack can
reach its destination — and every concrete weapon, ammunition type, and warhead
is configuration bound to one of them. Adding the fourteenth rifle, the third
plasma caster, or a tear-gas shell for the launcher touches `config.toml` and
nothing else. That is the whole design; the rest of this chapter is the
consequences.

---

## 1. The invariant: six archetypes, two terminals

Every attack, whatever its delivery, terminates in exactly two places:

1. **Units** — one or more `DamagePacket`s through the mechanics/06 pipeline
   (mitigation → quantized HP → events), plus statuses applied at the delivery
   site (a baton applies STUNNED where it connects; packets themselves stay
   damage-only).
2. **The world** — a **payload** executed through FieldEdit / the physics entry
   points: pressure, wall damage, gas emission, ignition, heat.

Weapons never invent a third path. Anything that reaches a unit goes through
mitigation; anything that reaches the world goes through the same choke-point
writes as every other system (engine/13, engine/14). This is what keeps combat
inside the determinism law and what makes "new weapon = new config row" true.

The six delivery archetypes (code — this set is closed until a genuinely new
physics of delivery appears):

| Archetype | Behavior | Covers |
|---|---|---|
| **HITSCAN** | marches its full range in one tick; can **skewer** several units and chew walls along the path | lasers |
| **PROJECTILE** | a marching entity with finite speed, swept segment per tick, stops on first wall/unit | bullets, plasma bolts, launcher rounds, shotgun pellets (×N per trigger) |
| **LOBBED** | projectile that ignores unit collision; detonates its payload on fuse expiry / arrival | hand grenades (the shipped `Projectile`) |
| **PLACED** | attached to a tile; detonates on schedule or trigger | door charge (shipped), C4 satchel |
| **SPRAY** | sustained cone of *field writes* over N ticks; no projectile entity at all | flamethrower, poison projector |
| **MELEE** | adjacency + the §3 to-hit/crit resolver | knife, arc baton |

The old draft's boundary rule survives as the spine: **kinetic damage travels on
the projectile/entity path; thermal/energy/chemical damage travels through the
fields.** A bullet finds a unit and applies a packet. A flamethrower never
touches a unit at all — it heats the world, and the existing heat coupling row
(mechanics/05) does the damage. SPRAY weapons are therefore almost free: the
Dragon-7 is *zero new damage code*, just aimed field writes.

---

## 2. Everything is a march

**Speed is data: tiles per tick.** All ranged delivery is one shared
tile-marcher; archetypes differ only in how far they march per tick and what
they do at tiles along the way.

- A **laser** marches its full range in one tick — physically instant, because
  photons are. No initiative table says so; physics does (the travel-time
  arbiter, mechanics/05 §4 P3).
- A **rifle round** at ~96 tiles/tick crosses any compartment in the tick it
  is fired — indistinguishable from hitscan at ship scales, but honest: at
  extreme range or against a fast mover, the round is genuinely in flight.
  Bullet speeds are authored ≥ the weapon's `range_tiles`, so small-arms fire
  resolves same-tick; the in-flight machinery is exercised by the slow
  archetypes.
- A **plasma bolt** at 1–2 tiles/tick is a slow, glowing, dodgeable projectile
  — and a transient light source for the ray engine.
- A **launcher round** at 1.25 tiles/tick (the shipped grenade `travel_speed`,
  30 tiles/s) arcs visibly across a room.

*(All per-tick speeds in this chapter are authored against the actual clock —
**24 ticks/second** — a W1 finding: the first draft assumed 12 tps and its
speed column was ×2 too high. `speed_tiles_per_tick × 24` is the tiles/second
of record.)*

A projectile that does not resolve within its tick persists as an in-flight
entity and continues next tick (the shipped grenade `Projectile` generalizes).
In-flight projectiles advance in **tick slot 2** (before movement), preserving
the causal pipeline ordering.

**Collision.** PROJECTILE marches tile-by-tile (the shipped bullet march):
first solid tile or first unit footprint stops it. LOBBED ignores units,
detonating at target/fuse. HITSCAN passes *through* units (skewer — each takes
a packet, attenuated §5) and decrements wall HP where it crosses solids.
Grenade bounce off walls stays out of v1 (straight line to target, the shipped
behavior); noted in §8.

---

## 3. The accuracy trinity

Three different questions, three mechanisms, no overlap. The principle
(mechanics/06 §5): **physics decides what is possible; probability models what
2D cannot see.**

**Spread answers distance.** Every shot's direction = aim bearing + an angular
offset drawn uniformly from the weapon's cone (`rng.uniform(-θ, +θ)` — door 4,
the shipped rifle pattern; trig through the deterministic kit). The bullet then
*physically flies*. Hit probability against a target of width `w` at distance
`d` is geometry, not a table: ≈ `min(1, w / (2·d·tanθ))`. One number per weapon
creates the close/mid/long classes: a 6° PDW is a coin flip at ~10 tiles, a
1.5° carbine at ~38, a 0.25° marksman rifle at ~230 — it simply does not miss
indoors. Missed shots are not deleted: they fly on and hit whatever is there —
walls (chewing them, §5), cover, or the second zombie in the file. Two spread
values per weapon in v1: `spread_deg` (aimed fire) and `spread_snap_deg`
(snap/auto-fire, movement); a continuous aim-time ramp is deferred (§8).

**The exposure roll answers cover.** A top-down ray overstates exposure — it
cannot see a crouched marine behind a crate. When the march would enter a
target's footprint having just crossed a tile whose material carries
`cover_exposure < 1.0` on that approach arc, the shot connects with probability
`cover_exposure` (× stance modifiers, later). **A shot that fails the roll is
absorbed by the cover tile** — its wall damage is deposited there, so
suppressive fire chews the crate until it stops *being* cover. Cover is
directional and flanking is geometric: attack from an arc the cover does not
protect and there is no roll at all. `cover_exposure` is a new **materials
table column** (default 1.0 = no concealment; crates/furniture ~0.5–0.6);
solid walls need no value — they stop the march physically.

**The crit roll answers facing.** On a connecting hit:
`crit% = weapon.crit_chance × arc multiplier` — ×1 front, ×2 flank, ×4 behind
(standard values). Arcs are computed from the target's synced facing with kit
trig; a crit multiplies the packet amount by `weapon.crit_mult` before
mitigation. Facing is universal but **arcs are data** (species profile): a
slime blob sets its vision arc to 2π and its flank/behind widths to zero — no
back to stab, zero special-case code. Melee runs through this same resolver
(the knife's `crit_chance` is high and its behind-arc work is the assassin
fantasy) — which is precisely the seam that lets this engine host an RPG later.

All three rolls consume the simulation's single seeded generator in fixed
order, so an entire firefight — covers, flanks, crits — replays bit-for-bit
from its seed. **Rolls are drawn lazily**: a roll that cannot matter is not
drawn at all (no cover on the approach → no exposure draw; `crit_chance = 0`
→ no crit draw). Draw count then depends only on synced state, so replay and
cross-machine identity hold, and a weapon with the feature dialed to zero
leaves the RNG stream — and therefore the golden digest — untouched (the
dormant-seam pattern, engine/06 ignition precedent).

---

## 4. The three tables

Data model (config.toml; loaded into id-indexed tables mirroring
`MaterialTable`/`GasTable`):

**`[weapons.<name>]`** — the delivery instrument.

| Column | Meaning |
|---|---|
| `archetype` | one of the six (§1) |
| `ammo_family` | what it feeds on (`"none"` for melee) |
| `spread_deg`, `spread_snap_deg` | the §3 cones |
| `range_m` | hard cap (energy/drag) in **physical meters** — the W6 convention (below); `range_tiles`, the march length, is DERIVED from it once at level load |
| `shots_per_trigger` | burst/pellet count per fire action |
| `rof_interval_seconds` | cadence gate between triggers |
| `mag_size`, `reload_seconds` | ammo economy (0 = not tracked, until W3 wires it) |
| `ap_cost` | order cost (turn system) |
| `crit_chance`, `crit_mult` | the §3 crit base |
| `mass_kg` | handling/encumbrance (future), melee impulse (now) |
| `loudness` | **reserved, no consumer yet** — emitted sound level 0..1 for the stealth layer (sound-hunting zombies, suppressed weapons). Data lands now so the armory is authored once. |
| `default_ammo` | (W6) the **static round-selection seam**: `""` = the W2 first-family-match; a row name = this weapon's standard round. Lets two weapons share a family honestly (the P12 and MP-11 both eat 9mm but load different rounds; the Lance-5 draws the heavy cell while the Lance-3 keeps the standard). Per-unit selection (loadout UI) stays §7. |

**Meter-based ranges (W6, Erik's decision 2026-07-07 — the conversion
convention of record).** Reach is authored as `range_m`, physical meters, so
a weapon covers the same distance whatever the grid resolution. At table
build (Simulation construction) each row derives its integer march length
from the loaded level's tile size:

    range_tiles = max(1, int(range_m / tile_size_m + 0.5))     # round-half-up

quantized ONCE (engine/14 door 2; the divide is one correctly-rounded IEEE
op, door 3) — every consumer (march length, fire-order range gate, spray
cone) keeps reading the integer `range_tiles`. The pinned test worlds are
1.0 m/tile, where meters and tiles coincide: every pre-W6 tile count became
the same number in meters, so effective ranges there are bit-identical and
the golden never moved. The playground (0.333 m/tile) derives 3× the tiles
for the same physical reach. Direct `WeaponDef(range_tiles=...)`
construction (the dict-table test path) remains valid; authoring both
columns on one config row is a loud load error. The ONE deliberate behavior
change: the Dragon-7's range became **10 m** (§6 — the W4 value was
authored pre-meters as 8 tiles and read as a 2.7 m sputter in the
playground: the W4 feel-check's "invisible flamethrower", now in numbers).

**`[ammo.<name>]`** — the round. `family` (must match the weapon's),
`damage`, `dtype` (mechanics/06 type), `ap` (armor pierce),
`speed_tiles_per_tick`, optionally `payload` (a payload row ref, for
explosive/gas rounds), and `glow` (W6, **render-only**: a nonempty profile —
`"plasma"` — makes the in-flight march emit one `ProjectileGlowEvent` per
tick; the renderer draws the glowing bolt + a transient light and the sim
never reads the column back). Swappable ammo is the point: AP rifle rounds,
incendiary shells, and late-game exotics are new rows here — the
progression hook.

**`[payloads.<name>]`** — what happens at the destination (executed via
FieldEdit / the physics entry points): `radius`, `pressure` (wave source),
`wall_damage`, `unit_damage` (BLAST packets with falloff),
`gas_species` + `gas_amount` + `gas_radius` (emission into the engine/05 gas
slices), `ignite_radius` + `ignite_intensity`, `heat_amount` + `heat_radius`
(W6 — a one-shot DISC heat deposit into the engine/06 `heat` ingress buffer
with linear falloff, `payloads.deposit_heat`: the plasma splash; converts
to temperature the SAME tick, so ignition and the heat|max unit-damage row
both come free — the SPRAY two-terminals discipline applied to a
detonation), and the smoke boolean SPLIT
(the W1 finding): `clear_smoke` (data-of-record — the inner-radius clear
lives inside `apply_explosion` in v1) + `emit_blast_smoke` (live — gates the
textured cloud). **Hand-grenade ammo rows and 40 mm launcher rounds point at
the same payload rows** — one definition of "frag", "smoke", "tear",
"poison", delivered by hand or by tube (row-object identity is a W3 test
gate). The shipped `apply_explosion` triple became the payload *executor*
(`simulation.payloads.execute_payload`, W3), with the grenade =
`payloads.frag_standard` and the door charge = `payloads.breach_focus`,
byte-for-byte (replica-proven).

---

## 5. Archetype details & determinism notes

- **HITSCAN (laser).** The old draft's beam design, kept whole: a serial weapon
  pass, before the lighting raycaster, marching the shot vector once. It
  skewers every unit on the line (ENERGY packet each), deposits heat + wall
  damage where it crosses solids (a sustained beam burns through), and its glow
  is *just light* — a transient source the normal lighting pass picks up, plus
  a muzzle-flash companion profile. **Blessed v1 feature: gas attenuates the
  beam.** Per tile the beam's energy is multiplied by
  `max(0, ONE − Σ absorb_g·density_g)` in Q16.16 — the same Beer-Lambert the
  renderer uses, but in integer per-tile multiplicative form: no `exp`, no
  transcendentals, door-1 arithmetic on the int32 gas field. Smoke grenades
  are thereby laser countermeasures, and no code knows it. Lasers are *quiet*
  (`loudness` ~0.1): the stealth-tech identity, countered by the cheapest
  grenade in the game.
- **PROJECTILE.** The shipped bullet march + per-shot spread, generalized:
  per-tick march length = `speed_tiles_per_tick`, in-flight persistence beyond
  it. Shotguns are `shots_per_trigger = 8` pellets on one trigger — **damage
  falloff emerges from pellet spread geometry**, no falloff table. Plasma
  bolts carry a small heat payload (splash + ignition) and a glow profile.
  - **Direct-hit rule (W6, of record):** a marching round applies its
    direct-hit packet iff its ammo row authors `damage > 0` — the payload
    presence no longer decides. Every kinetic small-arm (damage > 0, no
    payload) and the 40 mm (damage 0 + payload — the W3 "the blast does
    the work" rule) are bit-identical to before; a PLASMA bolt authors
    BOTH: it hits the unit it stops on for its HEAT packet (40/70 — the
    zombie ×4 HEAT vulnerability applies at mitigation, and the bullet
    site rule's `bullet_damage_multiplier` truncation applies before it,
    since plasma rides the bullet march) AND detonates its splash at the
    same stop tile through the executor (event kind `"shell"`).
  - **Plasma splash (W6, of record):** `plasma_splash_small/large` =
    modest `wall_damage` (25/45 — a SCORCH: wood walls carry hp 60, and a
    bolt must not destroy the partition it wants to ignite), a small
    pressure pop, an ignite ring, and the `heat_amount` one-shot disc
    (3200/5600 — centre temperature jump `heat/thermal_mass` = 400/700 on
    wood's mass 8, comfortably over its 300 threshold after the same-tick
    ×31/32 cool). `unit_damage` stays 0: the DIRECT HIT carries the unit
    damage; bystanders cook via the heat|max row reading the splash.
  - **The glow (W6):** `glow = "plasma"` on the ammo row → one render-only
    `ProjectileGlowEvent` per advanced tick (position ping); the renderer
    draws the ember + a transient light. At 1.5/1.25 t/t a bolt is
    genuinely watchable — the first in-game user of the W2 in-flight
    persistence machinery (`Simulation.bullets`).
- **LOBBED / PLACED.** The shipped grenade and door charge, re-homed onto
  weapon+ammo+payload rows. C4 = PLACED with a demolition payload (radius ~8,
  wall damage ~800 — "bigger bombs") and a trigger mode (timer or the shipped
  det-slot schedule); only player-issued charges detonate.
- **SPRAY** *(built by W4 — this bullet is the implementation of record).*
  N ticks of aimed cone field-writes (FieldEdit TILE edits, `combat.py`:
  `spray_cone_tiles` / `deposit_spray_cone` / `process_sprays`): the
  flamethrower deposits heat into the `heat` ingress buffer (the engine/06
  path: C++ TemperatureSolver convert → `temperature` → the live
  `apply_temperature_ignition` seam) + emits `fuel_gas` for the future
  per-gas combustion (engine/05 M3); units burn via the *existing* heat
  coupling row — zero new damage code, tested as an invariant. The poison
  projector emits the `poison` species through the identical code path —
  damage via the W3 `gas[poison]` row (no blindness: poison ≠ teargas) —
  and the alien's breath weapon is a config row.
  - **Cone-angle convention (of record):** the §6 armory quotes the FULL
    cone ("30° cone"); config authors `cone_half_angle_degrees` — the
    membership test's natural quantity — so the Dragon-7 row carries 15.0.
  - **Membership is pure integer** (door 1): a tile is in the cone iff
    `dot_q ≥ 0 ∧ dot_q² ≥ (dx²+dy²)·c_q²`, where `dot_q = dx·cos_q +
    dy·sin_q` on the kit aim bearing's Q16.16 unit vector and `c_q` is the
    kit cosine of the half-angle — the squared form of `dot(tile_dir,
    aim_dir) ≥ |tile_dir|·cos(θ½)`. No per-tile atan2, no float compare;
    fixed row-major traversal; the apex is never a member. The **nozzle
    rule**: the shooter's own footprint tiles are excluded (the jet
    projects beyond the operator — a flamer never cooks itself).
  - **Occlusion:** a cone tile receives deposits only if
    `gmap.has_los(shooter, tile)` — flames do not pour through walls. The
    Bresenham test passes a solid *endpoint* with a clear path, so the
    flame lands ON a wall face (how wood catches) but never beyond it.
  - **Falloff (documented form):** deposit = column ÷ `max(1,
    isqrt(dx²+dy²))` — a simple integer 1/distance falloff; one
    correctly-rounded IEEE divide (door 3), quantized ONCE per tile at the
    FieldEdit combine (door 2). Heat lands on solids (no skip-mask — walls
    catch); gas respects the solid skip + [0, 1] clamp of the gas policy.
  - **The WEGO trigger (v1 rules of record):** one trigger = one BURST =
    `burst_seconds` of consecutive deposit ticks (1.5 s → 36 @ 24 tps),
    deposited in the shooting slot (conductor step 4b). Spray fires only
    on an EXPLICIT fire order with no movement order in the same phase —
    **the sprayer stands still**; Move & Attack auto-fire SKIPS spray
    weapons. A standing order chains bursts back-to-back; `mag_size`
    counts BURSTS (Dragon-7: 4 per tank, 4 s swap — the W3 machinery
    unchanged). **Interruption:** composed `can_act` False stops the burst
    that tick, the fire order is consumed, no resume.
  - **Determinism:** the spray draws NO randomness anywhere — cone, aim,
    and falloff are kit/integer arithmetic, so a spray-free trajectory is
    bit-identical to pre-W4 (the dormancy replica gate).
  - **The jet visual (W6, of record):** every DEPOSITING tick also appends
    one render-only `SprayJetEvent` (apex / captured target / range /
    half-angle / kind) to the sim's tick events — the LaserFiredEvent
    precedent: emission is a pure function of already-synced state, the
    determinism digest hashes only UnitHit/UnitKilled, and the renderer
    draws a translucent cone fan + a transient warm light (`"flame"`) or
    a fainter sickly-green variant (`"miasma"`); it never writes back. An
    interrupted / finished burst emits nothing, so the hose vanishes with
    the flames. This closes the W4 finding "no spray visual yet".
- **MELEE** *(built by W5 — this bullet is the implementation of record).*
  Adjacency + the §3 resolver (`combat.py`: `melee_adjacent` /
  `melee_strike`, the melee branch of the `process_shooting` dispatch).
  The resolver collapses exactly as designed: **to-hit is trivially 1.0**
  — a strike happens at touching footprints, so there is no intervening
  tile to be cover and no march to absorb; the exposure roll does not
  EXIST on this path (never drawn — the lazy-roll rule, not an
  always-passing roll). The crit-vs-facing roll does the interesting work
  (the knife's 0.15 × the behind-arc ×4 = the assassin fantasy), drawn
  lazily through the same `attack_resolver` seams as bullets — a crit-0
  melee weapon (the baton) consumes ZERO randomness while swinging.
  - **The adjacency predicate (of record):** two units are melee-adjacent
    iff some occupied tile of one is within **Chebyshev distance 1** of
    some occupied tile of the other — 8-connected footprint contact: edge
    contact AND diagonal corner contact count, overlap counts trivially.
    Exact for any footprint shape (pairwise `occupied_tiles()`, not a
    bounding box). Deliberately NO `has_los` term: touching footprints
    have no tile between them to occlude (so diagonal corner contact
    across a wall corner CAN stab — accepted v1). Pure integer door-1
    arithmetic, paid per trigger attempt only.
  - **Trigger flow:** the fire order names a TILE (the shipped shape);
    the target is the first living enemy in stored unit order occupying
    it. Adjacency replaces the ranged range/LOS gates; the spread cone is
    meaningless on a blade (no cone draw, ever). A connecting strike
    charges the rof cadence (and the mag machinery — a no-op at the rows'
    mag 0); a **whiff charges nothing** and retries next tick while the
    order stands. Facing snaps to the strike bearing (the fire_burst
    rule). Auto-fire SKIPS melee (v1: an explicit order names the target
    tile; stab-on-contact Move & Attack is a §8 revisit).
  - **Packets stay damage-only; statuses at the delivery site:** the
    strike applies its `DamagePacket` (`melee_damage`/`melee_dtype` —
    weapon-row columns, melee feeds on no ammo) through the pipeline,
    then applies the row's `status_kind` (arc baton → STUNNED 1.5 s)
    SEPARATELY via `apply_status` — the W3 teargas→BLINDED pattern; the
    packet type has no status field to smuggle CC through. A killing
    blow applies no status (corpses don't get stunned). No zombie
    `bullet_damage_multiplier` — that is the BULLET site rule.
  - A shove impulse reusing the P4 `Δv = J/mass` push machinery is a
    natural v1.5. Zombie melee stays on its shipped `ai_zombie` path
    (regression-locked); migrating NPC attacks onto weapon rows is future
    work (§8) — the framework is team-agnostic by construction.

**Doors mapping (engine/14).** Weapon/ammo/payload numbers: door 2 (quantized
once where they feed synced state). Spread/exposure/crit rolls: door 4 (raw
stream: `uniform`/`integers` — affine on the bit-stream, no distribution
methods). All angles through the Q2-lift kit (`atan2/sin/cos_q16`). Beam
attenuation: door-1 integer arithmetic. New RNG consumers change replay
streams: **the golden digest moves at W2 and W3** — expected, re-baselined
with zero-field-movement proofs exactly like P3/P4.

---

## 6. The armory (standard values — Erik's tuning dials, not balance)

| Weapon | Class | Archetype | Family | Spread (aim/snap) | Trigger | Damage | Speed t/t | Loud |
|---|---|---|---|---|---|---|---|---|
| P12 "Whisper" | sidearm, suppressed | PROJECTILE | 9mm | 2.5°/5° | 1 @ 0.25 s | 12 KIN | 60 | 0.15 |
| MP-11 PDW | SMG, close | PROJECTILE | 9mm | 6°/9° | 4 @ 0.4 s | 7 KIN | 60 | 0.7 |
| K5 Carbine | assault rifle | PROJECTILE | rifle_556 | 3°/6° | 5 @ 0.5 s | 10 KIN | 96 | 0.8 |
| LR-50 | marksman/AM rifle | PROJECTILE | rifle_50 | 0.25°/2° | 1 @ 1.5 s | 90 KIN, AP 10 | 128 | 1.0 |
| Jackhammer-8 | shotgun | PROJECTILE | shell_12g | 8°/10° | 8 pellets @ 0.8 s | 6 KIN ×8 | 50 | 1.0 |
| Lance-3 | laser rifle | HITSCAN | cell_laser | 0.1° | 1 @ 0.5 s | 25 ENERGY (skewer) | ∞ | 0.1 |
| Lance-5 "Longlight" | heavy laser | HITSCAN | cell_laser | 0.05° | 1 @ 1.0 s | 55 ENERGY (skewer) | ∞ | 0.1 |
| Sunspot | plasma caster | PROJECTILE | cell_plasma | 1.5° | 1 @ 0.9 s | 40 HEAT + splash | 1.5 | 0.5 |
| Helios | heavy plasma | PROJECTILE | cell_plasma | 2° | 1 @ 1.4 s | 70 HEAT + splash | 1.25 | 0.6 |
| Dragon-7 | flamethrower | SPRAY | fuel_tank | 30° cone, range **10 m** (the W6 rescale) | 1.5 s burst | heat writes (2400/tick) | — | 0.6 |
| Dragon-9 heavy | heavy flamethrower | SPRAY | fuel_tank | 30° cone, range **20 m** | 2.0 s burst | heat writes (4800/tick) | — | 0.7 |
| Miasma Vent | poison projector | SPRAY | toxin_tank | 25° cone, range 7 m | 1.5 s burst | poison gas | — | 0.4 |
| Hand grenade | thrown | LOBBED | hand_grenade | — | fuse 0–10 s | payload | 1.25 | payload |
| GL-6 Revolver | grenade launcher | PROJECTILE | 40mm | 3° | 1 @ 1.2 s | payload | 1.25 | 0.9 |
| Breach charge | demolition | PLACED | demo_charge | — | det slot | `breach_focus` | — | 1.0 |
| C4 satchel | demolition | PLACED | demo_charge | — | timer/remote | `demolition_c4` | — | 1.0 |
| Combat knife | melee | MELEE | none | — | 1 @ 0.6 s | 35 KIN, crit 15 % | — | 0.05 |
| Arc baton | melee | MELEE | none | — | 1 @ 0.8 s | 10 ENERGY + STUNNED 1.5 s | — | 0.2 |

*(All row ranges are authored `range_m` since W6 — §4 conversion convention.
The full set is data in config.toml: every row above is loaded, validated,
and reachable in-game through the playground weapon-cycle key.)*

Payload rows: `frag_standard` (the shipped grenade: radius 5, pressure 10,
wall 200, unit 60), `breach_focus` (the shipped charge: 3/5.0/500/60),
`demolition_c4` (8/25.0/800/150), `smoke_screen` (white_smoke), `tear_burst`
(teargas), `poison_cloud` (poison), `incendiary_splash` (ignite ring),
`plasma_splash_small/large` (W6 — §5 PROJECTILE notes: scorch wall damage +
one-shot heat disc + ignite ring, unit_damage 0). The 40 mm ammo rows
(`40mm_frag/_smoke/_tear/_poison/_incendiary`) reference these same rows —
`grenade_incendiary` and `40mm_incendiary` share `incendiary_splash`
(row-object identity, the W3 gate pattern, re-pinned at W6). Ammo-family
sharing is deliberate where realistic: the P12 and MP-11 both eat 9mm — and
load different rounds through `default_ammo` (`9mm_subsonic` 12 KIN vs
`9mm_fmj` 7 KIN), as do the two Lances (`cell_laser_standard` 25 /
`cell_laser_heavy` 55), the two plasma casters, and the two Dragons
(`fuel_standard` 2400 / `fuel_heavy` 4800).

Gas grenade *effects* ride the coupling table (mechanics/05), not the payload:
`teargas` density → `BLINDED` (the `can_aim` consumer, shipped W3 — teargas
blinds: aimed fire collapses to the snap cone), `poison` density → POISON DoT
packets (zombies immune — they don't breathe). The payload only puts gas in
the air; what gas does to lungs is the exchange layer's job, same as heat.

---

## 7. Reserved & deferred (with intent)

- **ACID — reserved damage type, deferred with a love letter.** Alien-blood
  acid fits this game unusually well: a payload/SPRAY chemical that damages
  *floors* (material erosion), opening multi-z consequences — acid melts the
  deck, water pours down, fire below meets fuel above. Wants multi-level maps
  + a floor-HP analog first. `ACID` joins `ELECTRIC`/`PSY` on the mechanics/06
  reserved list now so no table needs renumbering later.
- **Grenade bounce** (straight-line v1, shipped behavior), **aim-time ramp**
  (two spread values v1), **suppression as a mechanic** (loudness + morale,
  far future), **NPC weapons on weapon rows** (zombie melee migrates when a
  second NPC weapon exists — the Miasma-armed alien is the natural trigger),
  **full inventory/encumbrance** (`mass_kg` waits for it), **electrical arc**
  (the old draft's conductor-seeking bolt — rides the engine/11 electricity
  chapter).
- **Ammo economy details**: v1 wires mags/reload/selection (W3); weight of
  carried ammo, scavenging, and the late-game exotic-ammo progression ride the
  mission/economy layer (agenda 6).

---

## 8. Implementation status

**Shipped (pre-framework draft):** rifle burst (cone, kit trig, march,
`has_los` gate, auto-fire, zombie `bullet_damage_multiplier` site rule),
grenades (`Projectile` + fuse + AP/count economy), door charges (det slots),
the explosion triple (`apply_explosion` / `apply_blast_damage` /
`add_explosion_smoke`), melee-by-zombie + conversion, event/tracer plumbing,
KINETIC packets through the mechanics/06 pipeline.

**Designed here, wired by the W-wave (plan of record, 2026-07-05):**

| Patch | Contents | Gate |
|---|---|---|
| **W1** | weapon/ammo/payload tables + loaders; re-home rifle→`k5_carbine`, grenade→`hand_grenade`+`frag_standard`, charge→`breach_charge`+`breach_focus`; `unit.weapon_id` | ✅ **SHIPPED** `2abf7dc` (2026-07-05): 521 green, golden `07c3f370…` byte-identical |
| **W2** | unified march (speed as data, in-flight persistence); spread aim/snap; §3 exposure/cover (+`cover_exposure` materials column) + crit/facing resolver; **Lance-3 laser** (skewer, wall-chew, integer gas attenuation, beam event) | ✅ **SHIPPED** `bbfb26a` (2026-07-05): 559 green (+55 W2 tests), golden `07c3f370…` **UNCHANGED** — the canonical scenario fires no weapon and every W2 roll is lazy, so stream and fields never move (findings below). Beam **glow-as-light deferred** to the explosion-light pass (`LaserFiredEvent` ships; the raycaster hookup lands with transient light sources) |
| **W3** | payload executor generalizing the explosion triple; gas payloads (smoke/tear/poison) + coupling rows (teargas→aim status, poison→DoT); GL-6 + 40 mm ammo; C4; ammo economy (mags/reload) | ✅ **SHIPPED** (2026-07-05): 587 green (+28 W3 tests), golden `07c3f370…` **UNCHANGED** — W3 adds **no RNG consumers anywhere** (the gas deposit is deliberately noise-free, the coupling rows are threshold-deterministic and take no generator, launcher/C4 reuse existing draw sites) and every new path is dormant in the canonical scenario. Byte-identity replica gates: frag+breach detonations AND a full scripted shipped-weapons round vs the verbatim pre-W3 site body (fields + events + RNG end-state). Findings below |
| **W4** | SPRAY: Dragon-7 + Miasma Vent (aimed sustained FieldEdit cones) | ✅ **SHIPPED** (merged `5594650`, 2026-07-07): 588 green (+15 W4 tests), golden `07c3f370…` **UNCHANGED** (W4 draws zero RNG by construction — kit/integer cone, deterministic falloff). Findings below |
| **W5** | MELEE: knife + arc baton through the resolver; STUNNED wiring | ✅ **SHIPPED** (merged `8f493a8`, 2026-07-07): 606 green (+17 W5 tests), golden `07c3f370…` **UNCHANGED** (the fifth patch running: melee is a dead branch in a melee-free scenario and a crit-0 melee weapon draws nothing even while swinging). Findings below |
| **W6** | METER-BASED ranges + the full armory as data + plasma detonate-at-stop + the spray-jet/plasma-glow visuals + the weapon-cycle debug key + standard-values audit | ⏳ **BUILT on branch `weapons-w6-armory`** (2026-07-08), **awaiting Erik's grand tuning session**: 626 green (+20 W6 tests, incl. the canonical-golden + untouched-RNG replica), golden `07c3f370…` **UNCHANGED** (the sixth patch running: meter conversion is identity at the pinned worlds' 1.0 m/tile, every new row is dormant until equipped, the new events are render-only). Findings below |

**W1 findings of record** (carry into later patches):

- **The clock is 24 tps** — this chapter's speed numbers were corrected
  accordingly (§2 note). `speed_tiles_per_tick × 24 = tiles/second`; the
  grenade row locks `1.25 × 24 == 30.0` (the shipped `travel_speed`) in a test.
- **W3 must split the smoke booleans**: today the door charge *and* the
  grenade both clear smoke (inner blast hole, inside `apply_explosion`) *and*
  emit the textured cloud (`add_explosion_smoke`). The payload executor needs
  `clear_smoke` + `emit_blast_smoke` as separate columns, both `true` on
  `frag_standard` *and* `breach_focus`, or the door charge silently loses its
  smoke on the day the executor takes over.
- `max_throw_range` had **no consumer** in the shipped code (dead config) —
  carried onto `hand_grenade` as data-of-record; a real throw-range check can
  wire in W3/W6.
- Weapons tables are **construction-bound** (rebuilt per `Simulation`
  reset/restart, not Ctrl+R) — matching materials/gases, engine/12 §5.

**W2 findings of record** (carry into later patches):

- **The golden did NOT move at W2** — §5's "the golden digest moves at W2"
  prediction was wrong in the good direction: the canonical A/B scenario
  contains no firing, and the lazy-roll rule keeps every new consumer
  (exposure, crit) off the RNG stream unless cover/crit genuinely engage.
  Chew only moves `wall_hp` where bullets actually stop. Verified before and
  after the wave: aggregate `07c3f370…` bit-identical. W3's gas payloads
  will be the first genuinely golden-moving patch *if* the scenario gains a
  detonation — same rule: re-baseline only with per-field proofs.
- **`destroy_wall` gate widened** (`solid` → `material != MAT_AIR`) so
  bullet chew can break a crate — cover that stops being cover. Audit
  finding riding along: the C++ fire burn-through list is `is_wall`-gated,
  so **fire never destroys furniture tiles today** (it depletes their
  `wall_hp`/fuel but the tile survives at 0) — a seam for the fire system
  to claim when burning furniture should collapse.
- **Facing already existed** (Q2-lift: kit `atan2` in `face_towards`,
  hashed as `__unit_facing__`); W2 only added the fire-time update (facing
  = aim bearing, before spread). Digest surface unchanged.
- **HITSCAN ignores cover in v1** (deliberate, with the no-crit rule):
  skewer + attenuation is its identity — a beam cannot be "absorbed by a
  crate" without breaking the pass-through contract. Revisit only if soft
  cover should attenuate beams (then as a material *attenuation* column,
  not an exposure roll).
- **Ammo resolution is first-family-match** (`ammo_for_weapon`) until the
  W3 economy wires real selection — one standard round per family holds
  until then.

**W3 findings of record** (carry into later patches):

- **The golden did NOT move at W3 either** — §5's prediction fell the good
  way twice: W3 was engineered to add **zero RNG consumers** (gas deposits
  are deterministic no-noise discs; the coupling rows take no generator;
  the GL-6/C4 reuse existing draw sites), so with the canonical scenario
  throwing no gas the stream and every field trajectory are untouched.
  Verified: aggregate `07c3f370…` bit-identical before/after.
- **Gas transport verification (engine/05 §6.2, checked in code):** the C++
  `run_substeps` per-gas loop steps **ALL N slices** every tick (each
  non-empty plane on the shared wind with its own `[gases.*]` diffusion;
  empty planes skip via an exact integer `.any()`), so smoke_screen /
  tear_burst / poison_cloud clouds genuinely advect + diffuse — **no
  transport gap**. (Per-gas `decay` remains loaded-not-applied, as
  documented there.)
- **The can_aim consumer landed** (the owed P3 seam): composed `can_aim`
  False (teargas `BLINDED`, STUNNED, PARALYZED) collapses an aimed fire
  order to `spread_snap_deg` — one gate at cone selection in
  `process_shooting`.
- **Detonation semantics of record:** a payload PROJECTILE (40 mm) applies
  **no direct-hit packet** (`damage = 0`; the blast does the work) and
  detonates at its stop tile — first solid (the blast centres ON the wall
  tile, like a charge on a door), footprint entry, cover absorption, or
  max-range airburst. Event kind `"shell"`.
- **Mag state lives on the unit** (`current_mag` / `reload_done_tick`),
  deliberately **outside the synced digest surface** — matching the
  `last_fire_tick` precedent (combat-cadence state is a deterministic
  derivation of synced inputs, not hashed; a divergence surfaces one tick
  later in the hashed hp/event stream). The **round boundary tops
  magazines off** (v1 rule + the tick-rewind correctness twin of
  `last_fire_tick = -999`). Decrement is per TRIGGER; the auto-reload
  stall starts when the mag empties; no manual reload order in v1.
- **Immune units draw no DoT packet** (poison row, lazy-emission rule): a
  `resist_mult[POISON] == 0` unit (zombies — they don't breathe) is
  skipped entirely — no 0-damage `UnitHitEvent` spam on a horde standing
  in gas. Resistances strictly between 0 and 1 still compose through the
  pipeline.
- **Manual reload / per-type grenade loadout UI = W6** (`has_grenade` /
  `has_explosive` stay the single count pools; orders carry `ammo_name`,
  `None` = the shipped defaults).

**W4 findings of record** (carry into later patches):

- **The golden did not move at W4 either** — the fourth patch running: W4
  is RNG-free BY CONSTRUCTION (no spread on a cone weapon; membership,
  aim, and falloff are kit/integer arithmetic; the FieldEdits carry
  ``noise = 0``), so even a *firing* flamethrower leaves the RNG stream
  untouched. Verified: aggregate `07c3f370…` bit-identical, plus a
  full-sim dormancy replica (spray pass no-opped vs live).
- **The heat lands in `gmap.heat`** — the same Q16.16 ingress buffer the
  fire heat-rays and the laser feed: FieldEdit `field="heat"` TILE ADDs,
  flushed at conductor 6b BEFORE physics, so the C++ TemperatureSolver
  converts this tick's flame into `temperature` this same tick and
  `apply_temperature_ignition` (9d) reads it. Both existing consumers come
  free: ignition AND the heat|max unit-damage row (9c) — the two-terminals
  invariant holds with zero new damage code (tested).
- **`heat_deposit = 400` derivation** (in the config row, of record): wood
  (`thermal_mass` 8 → convert `D/8`, `COOL_SHIFT` 5 → ×31/32 per tick)
  under a sustained falloff-scaled deposit reaches `T_∞ ≈ 3.875·D`; wood's
  300 threshold falls at ~7 / ~15 / ~27 ticks for dist 1 / 2 / 3 —
  measured live at exactly 15 and 27. Beyond ~5 tiles the direct jet alone
  stays sub-ignition; the fires it starts cascade outward by their own
  radiation (`k_fire_heat`).
- **The nozzle rule** (new, forced by geometry): the shooter's 3×3
  footprint overlaps every cone's distance-1 ring, so without excluding
  the shooter's own tiles the heat|max row would cook the operator. The
  cone therefore skips the shooter's footprint — documented in §5.
- **The marine's weapon went data-driven** (`[marine] weapon`, consumed at
  Unit construction — W1's code literal deleted). Construction-bound like
  the tables: config edit → RESTART; Ctrl+R re-arms nothing (engine/12 §5).
- **Spray burst state lives on the unit** (`spray_ticks_left` /
  `spray_target` / `spray_order`), outside the synced digest surface — the
  mag-state precedent; cleared at the round boundary (the tick-rewind
  hazard twin of `last_fire_tick = -999`).
- **No spray visual yet**: the burst emits no tracer/event — the feel-check
  reads the CONSEQUENCES (ignition, fire glow, gas overlays, the T debug
  overlay). A flame-jet visual is a natural W6/renderer item if Erik wants
  the hose itself visible.

**W5 findings of record** (carry into later patches):

- **The golden did not move at W5 either** — the fifth patch running, and
  the strongest form yet: not only is melee dormant in a melee-free
  scenario (a dead archetype branch; the dormancy replica pins
  bit-identity against a sentinel-patched pre-W5 dispatch), but a crit-0
  melee weapon draws NOTHING even while actively swinging (the e2e
  chain-stun test runs a whole fight on an untouched RNG stream). The
  only melee RNG consumer is the knife's lazy crit draw — exactly one
  door-4 uniform per connecting strike.
- **The adjacency predicate is Chebyshev-1 footprint contact** (§5, of
  record): 8-connected, diagonal corners count, exact for any footprint
  via pairwise `occupied_tiles()`. It deliberately differs from the
  zombie bite's shipped center-distance rule (`footprint + 1` Euclidean)
  — two predicates, two paths, both pinned; they unify if/when zombie
  melee migrates onto weapon rows (§7).
- **Melee damage lives on the weapon row** (`melee_damage` /
  `melee_dtype` + the `status_kind`/`status_seconds` delivery-site
  columns): melee feeds on no ammo (`ammo_family = "none"`), so the
  packet numbers had nowhere else to live. Load-time validation is loud:
  a melee row must author damage + dtype; a status needs a positive
  duration; names resolve against the mechanics/06 registries.
- **No `bullet_damage_multiplier` on melee** — the site rule stays a
  bullet-path artifact (mechanics/06): the knife hits a zombie for its
  plain mitigated 35, pinned in the gate. Erik's tuning session (W6) is
  where knife-vs-horde balance gets its pass.
- **A killing blow applies no status** — corpses don't get stunned:
  statuses freeze on corpses (mechanics/06 §4), so a corpse status would
  be dead weight in the `__unit_status__` digest surface forever.
- **Chain-stun is real** (e2e-pinned): the tick order (statuses →
  shooting → zombie AI) means a baton strike stuns an adjacent zombie
  BEFORE its bite that same tick, and the 19-tick cadence re-stuns
  inside every 36-tick window — one marine with a stick can hold one
  zombie indefinitely (and takes 10 hp/strike off it). Whether that is a
  feature or a tuning problem is Erik's W6 call (`status_seconds` vs
  `rof_interval_seconds` is the dial).

**W6 findings of record** (carry into the wave close + Erik's session):

- **The golden did not move at W6 either** — the sixth patch running, and
  the widest surface yet: the meter→tile conversion is the identity map on
  the pinned 1.0 m/tile worlds (every derived `range_tiles` equals the old
  authored int, bit-for-bit); every new armory row is DORMANT until a unit
  equips it; the new events are render-only and outside the digest surface
  (verified, not assumed — the harness hashes UnitHit/UnitKilled only);
  and the W6 gate pins BOTH the canonical 30-tick aggregate digest AND the
  untouched fresh-seed RNG end-state (the scenario draws zero randomness).
- **The conversion convention** (§4): `max(1, int(range_m / tile_size_m +
  0.5))`, derived once at `Simulation` construction (`rebuild_tables(
  tile_size_m=gmap.tile_size_m)`); a bare `get_tables()` binds 1.0 m/tile —
  the test-world convention. Round-half-up, floor at 1 tile.
- **The direct-hit rule generalized** (§5): packet iff `ammo.damage > 0`
  (was: iff no payload). Bit-identical for every pre-W6 row; it is what
  lets plasma hit AND splash. Riding along: the zombie
  `bullet_damage_multiplier` site rule applies to plasma's HEAT hit (it is
  the bullet march), then the ×4 HEAT resist re-inflates it at mitigation
  — 40 → 10 → 40: a wash on zombies by coincidence of the standard values;
  Erik's call whether that composition stays.
- **`default_ammo` landed as the static round-selection seam** — the §6
  armory demanded per-weapon rounds inside shared families (P12/MP-11,
  the Lances, the plasmas, the Dragons); first-family-match stays the
  fallback and every pre-W6 weapon resolves exactly as before. Per-UNIT
  selection is still the §7 loadout item.
- **The W4 structural pin evolved**: `process_sprays` now takes `events`
  (the SprayJetEvent emission) — the no-unit-damage invariant is held by
  the runtime proof (a full burst emits ONLY jet events, no packet, no HP
  movement) plus `deposit_spray_cone` staying rng/events-free.
- **Plasma splash wall damage must stay under the wood wall's 60 hp**
  (25/45 authored): the first cut (60) DESTROYED the wood face on impact —
  the bolt blew away the wall it was supposed to ignite. Found by the
  ignition e2e gate; the dial comment carries the warning.
- **The Dragon-9's cone pass is the priciest trigger in the game** (O(r²)
  Python membership + per-tile Bresenham at 60 playground tiles per deposit
  tick); playable, and the vectorized cone is the optimization seam if
  continuous hosing drags.
- **The weapon-cycle key cycles TRIGGERABLE rows only**
  (`FIRE_ORDER_ARCHETYPES` = projectile/hitscan/spray/melee): LOBBED and
  PLACED have no trigger path (their order modes are G and B), and a
  breach-charge round has no march speed — cycling onto it would arm a
  weapon that cannot fire (and the demo round would crash the march).
  Deviation from the letter of "all [weapons.*] rows", by construction.

**Not built / explicitly owed:** everything in §7; the exposure/crit numbers
are standard values pending Erik's playground pass (his W6 tuning session is
the scheduled venue — chain-stun cadence vs stun duration explicitly his
call, numbers untouched at W6); beam glow-as-light for the LANCE lines (the
explosion-light pass — the spray-jet and plasma-bolt transient lights landed
at W6, the beam's own raycaster hookup still pends); per-unit ammo SELECTION
UI (W3 wired mags/reload, W6 wired per-weapon default rounds; round choice
mid-mission pends the loadout pass).
