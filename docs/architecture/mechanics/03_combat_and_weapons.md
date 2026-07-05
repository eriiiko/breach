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
| `range_tiles` | hard cap (energy/drag; the march length) |
| `shots_per_trigger` | burst/pellet count per fire action |
| `rof_interval_seconds` | cadence gate between triggers |
| `mag_size`, `reload_seconds` | ammo economy (0 = not tracked, until W3 wires it) |
| `ap_cost` | order cost (turn system) |
| `crit_chance`, `crit_mult` | the §3 crit base |
| `mass_kg` | handling/encumbrance (future), melee impulse (now) |
| `loudness` | **reserved, no consumer yet** — emitted sound level 0..1 for the stealth layer (sound-hunting zombies, suppressed weapons). Data lands now so the armory is authored once. |

**`[ammo.<name>]`** — the round. `family` (must match the weapon's),
`damage`, `dtype` (mechanics/06 type), `ap` (armor pierce),
`speed_tiles_per_tick`, and optionally `payload` (a payload row ref, for
explosive/gas rounds). Swappable ammo is the point: AP rifle rounds, incendiary
shells, and late-game exotics are new rows here — the progression hook.

**`[payloads.<name>]`** — what happens at the destination (executed via
FieldEdit / the physics entry points): `radius`, `pressure` (wave source),
`wall_damage`, `unit_damage` (BLAST packets with falloff),
`gas_species` + `gas_amount` + `gas_radius` (emission into the engine/05 gas
slices), `ignite_radius` + `ignite_intensity`, and the smoke boolean SPLIT
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
- **LOBBED / PLACED.** The shipped grenade and door charge, re-homed onto
  weapon+ammo+payload rows. C4 = PLACED with a demolition payload (radius ~8,
  wall damage ~800 — "bigger bombs") and a trigger mode (timer or the shipped
  det-slot schedule); only player-issued charges detonate.
- **SPRAY.** N ticks of aimed cone field-writes (FieldEdit): the flamethrower
  deposits heat (the engine/06 temperature path handles ignition — the
  `apply_temperature_ignition` seam is already live) + emits `fuel_gas` for the
  future per-gas combustion (engine/05 M3); units burn via the *existing* heat
  coupling row — zero new damage code. The poison projector emits the `poison`
  species through the identical code path — the alien's breath weapon is a
  config row. A SPRAY trigger in WEGO = a fire order sustained for
  `burst_seconds` of ticks.
- **MELEE.** Adjacency + the §3 resolver (to-hit vs exposure is trivially 1.0
  without cover; crit arcs do the work). Packets as usual; statuses at the
  site (arc baton → STUNNED); a shove impulse reusing the P4 `Δv = J/mass`
  push machinery is a natural v1.5. Zombie melee stays on its shipped
  `ai_zombie` path for now; migrating NPC attacks onto weapon rows is future
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
| Dragon-7 | flamethrower | SPRAY | fuel_tank | 30° cone, range 8 | 1.5 s burst | heat writes | — | 0.6 |
| Miasma Vent | poison projector | SPRAY | toxin_tank | 25° cone, range 7 | 1.5 s burst | poison gas | — | 0.4 |
| Hand grenade | thrown | LOBBED | hand_grenade | — | fuse 0–10 s | payload | 1.25 | payload |
| GL-6 Revolver | grenade launcher | PROJECTILE | 40mm | 3° | 1 @ 1.2 s | payload | 1.25 | 0.9 |
| Breach charge | demolition | PLACED | demo_charge | — | det slot | `breach_focus` | — | 1.0 |
| C4 satchel | demolition | PLACED | demo_charge | — | timer/remote | `demolition_c4` | — | 1.0 |
| Combat knife | melee | MELEE | none | — | 1 @ 0.6 s | 35 KIN, crit 15 % | — | 0.05 |
| Arc baton | melee | MELEE | none | — | 1 @ 0.8 s | 10 ENERGY + STUNNED 1.5 s | — | 0.2 |

Payload rows: `frag_standard` (the shipped grenade: radius 5, pressure 10,
wall 200, unit 60), `breach_focus` (the shipped charge: 3/5.0/500/60),
`demolition_c4` (8/25.0/800/150), `smoke_screen` (white_smoke), `tear_burst`
(teargas), `poison_cloud` (poison), `incendiary_splash` (ignite ring). The
40 mm ammo rows (`40mm_frag/_smoke/_tear/_poison`) reference these same rows.
Ammo-family sharing is deliberate where realistic: the P12 and MP-11 both eat
9mm.

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
| **W4** | SPRAY: Dragon-7 + Miasma Vent (aimed sustained FieldEdit cones) | **HUMAN-TEST** — Erik feel-checks before merge |
| **W5** | MELEE: knife + arc baton through the resolver; STUNNED wiring | suite green |
| **W6** | armory playground room + weapon-cycle debug key + full standard-values audit | **HUMAN-TEST** — Erik's tuning session |

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

**Not built / explicitly owed:** everything in §7; heat-damage tuning vs the
armory numbers; the exposure/crit numbers are standard values pending Erik's
playground pass; beam glow-as-light (the explosion-light pass); ammo
SELECTION UI (W6 — W3 wired mags/reload; per-unit round choice pends the
loadout pass).
