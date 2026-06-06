# Combat & Weapons

**Depends on:** [Units & Entities](01_units_and_entities.md), [Ray Engine](../engine/08_ray_engine.md)

Combat in Breach is two cooperating layers. The first is **direct combat** —
bullets, melee, line-of-sight — which is fast, serial, and resolved per tick
against the unit list. The second is **combat-via-physics**: weapons that do not
hit a unit so much as alter the world the unit stands in. A grenade does not
"deal area damage"; it deposits pressure, heat, and wall damage into the
atmosphere and structural fields, and whatever those fields then do to the units
is the damage. This split is deliberate. It keeps the things that must be cheap
and exact (a bullet hitting a target) on a simple deterministic path, and routes
everything spatial and emergent (blasts, fire, beams) through the same shared
physics that already runs every tick.

All combat lives behind the deterministic `Simulation` facade. Nothing here
reaches into raw fields ad hoc: weapons mutate the world only through a small set
of physics entry points (`apply_explosion`, `add_explosion_smoke`) and read it
only through `gmap.<field>` and `gmap.has_los`. Every source of randomness — the
bullet cone, explosion-smoke texture — is drawn from the simulation's single
seeded `numpy.random.Generator`, so an entire firefight replays bit-for-bit from
its seed. This is what makes combat usable as training data and as a regression
fixture, not just as a game.

---

## The two combat layers

| Layer | Mechanism | Determinism source | Damage path |
|---|---|---|---|
| Direct (bullets, melee) | ray-march / adjacency against unit list | serial CPU loop; cone offsets from seeded RNG | written straight to `unit.current_hp` |
| Via-physics (explosions, beams, fire) | deposit into pressure / heat / wall-hp fields | integer field math; field then sampled serially | derived from the field, applied in unit logic |

The boundary between them is the rule that **kinetic damage travels on the
projectile/entity path, and thermal/energy damage travels through the fields.**
A bullet is an instantaneous ray that finds a unit and subtracts HP. A beam,
explosion, or fire never writes a unit inside a kernel — it changes the world,
and the change is sampled afterward by serial, deterministic unit logic. This is
the same principle the ray engine enforces (the deposit-only lighting kernel
never writes units; units sample the buffers after the pass).

---

## Direct combat

### The rifle

The rifle is the baseline weapon every marine carries. It fires in **bursts**:
five bullets per burst, with a short cadence gate (`burst_interval_ticks`)
between bursts so a unit can't empty into a target every tick.

Each bullet is an independent ray. Its angle is the shooter→target bearing plus a
random offset sampled uniformly from the weapon's cone half-angle (3°). The ray
marches tile-by-tile out to the weapon's range, stopping at the first wall or the
first unit it enters:

```
base_angle = atan2(target - shooter)
for each of bullets_per_burst:
    angle = base_angle + rng.uniform(-cone, +cone)   # the determinism site
    march from shooter along angle, up to range_tiles:
        if tile is_wall:            stop (miss into wall)
        if tile inside a unit box:  stop, that unit is hit
    if hit: apply damage; append a Shot tracer
    else:   append a Shot tracer to the stop point anyway
```

The per-bullet cone is the single nondeterministic decision in rifle fire, which
is exactly why it draws from the simulation RNG rather than process-global
`random`. Every bullet — hit or miss — emits a `Shot` tracer (a purely visual,
caller-owned effect, not gameplay state) and a `ShotFiredEvent` for the renderer
to animate, plus `UnitHitEvent` / `UnitKilledEvent` on a hit.

**Damage scaling.** A bullet does flat damage to marines. Against zombies it is
scaled by `zombie.bullet_damage_multiplier` (0.25) — small-arms fire is poor at
stopping them, which is the core combat tension of the game. (Fire, by contrast,
is devastating to zombies via `fire_damage_multiplier`; that coupling lives in
the fire system, but it is why explosives and incendiary tactics matter more than
trigger discipline.)

**Targeting modes.** A unit fires either because it has an explicit fire order
for the current phase, or because it is executing a Move & Attack order, in which
case it **auto-fires** at the nearest visible enemy within range each burst
interval. Both go through the same range check, line-of-sight check, and burst.

### Line of sight

Targeting is gated by `gmap.has_los(observer, target)` — a ray from the shooter
center to the target that stops at the first wall. Today this is a Bresenham
walk over `is_wall`; it is deliberately exposed as an **interface** rather than
inlined, so the backing implementation can be upgraded (and so "can this unit
see heat through smoke?" becomes the same query against the heat channel —
infravision — without touching call sites). A shot is only attempted when both
the range check and the LoS check pass.

### Melee

Melee is the zombies' weapon. A zombie that is adjacent to a marine (within
`footprint + 1` tiles) attacks on a cooldown (`attack_cooldown_ticks`), dealing
`melee_damage`. The critical side effect is the kill flag: a marine killed by
melee is marked `killed_by_zombie`, and at end-of-round every such marine
**converts** into a zombie, keeping its inventory (so a converted marine that
still carries a grenade can cook it off later). Explosion and bullet deaths do
**not** set this flag — only melee converts. This is what turns a single
breached room into a cascade.

---

## Combat via physics: explosions

An explosion is the archetype of the via-physics layer. Grenades and door
charges both detonate through one shared entry point, `apply_explosion`, which
makes a single coordinated edit to several fields at once — pressure, walls,
fire, smoke — because a real blast does all of these together and they must stay
consistent.

For every tile within the radius (with linear distance falloff):

| Effect | Field touched | Behavior |
|---|---|---|
| Structural damage | `wall_hp` | `-= wall_damage × falloff`; tile destroyed via `destroy_wall` at HP 0 |
| Shockwave | `wave_source` | pressure deposited through a smoothed **3×3 kernel** (weights 4/2/1, normalized /16) so the wave equation gets a clean, non-spiky source |
| Sustained wind | `atmosphere` | direct pressure boost — the lingering gradient that drives smoke and venting |
| Smoke clearing | `smoke` | zeroed in the inner 40 % of the radius (the blast punches a hole in the cloud) |
| Ignition | `fire` | flammable tiles inside 70 % of the radius ignited at `0.5 × falloff` |

The 3×3 smoothing on the shockwave source, and a per-tick `max_source_per_step`
rate limiter in the atmosphere solver, exist because a raw point deposit of a
large pressure spike makes the wave solver ring or destabilize. Smearing the
source across a small kernel keeps the IMEX pressure step well-behaved while
still producing a sharp, expanding shock.

Notably, `apply_explosion` writes **no unit damage**. Damage to units is a
separate, explicit call (`apply_blast_damage`) — the kinetic part of a blast,
kept on the serial unit path. It damages every live unit within the radius with
linear falloff, but only if the result clears `blast_damage_threshold` (5),
which suppresses meaningless chip damage at the edge of distant blasts. Blast
kills do not set `killed_by_zombie`. A third call, `add_explosion_smoke`, lays
down the textured smoke cloud (per-tile noise drawn from the RNG).

### Grenades

A grenade is a thrown **projectile** with a fuse. When execution begins, queued
grenade orders materialize as in-flight `Projectile` objects that travel in a
straight line from thrower to target at `travel_speed`, detonating when the tick
reaches `thrown_tick + fuse × ticks_per_second`. The fuse is player-set (0–10 s,
default ~1 s), which is what lets a marine cook a grenade to airburst it among
moving zombies or bank it to detonate next phase. At detonation the projectile
fires the standard explosion triple — `apply_explosion`, `apply_blast_damage`,
`add_explosion_smoke` — and emits an `ExplosionEvent`. Undetonated long-fuse
grenades carry across the round boundary.

Issuing a grenade order costs AP and decrements the unit's `has_grenade` count;
canceling the order refunds both. This is the seed of the planned ammunition
model (below) — grenades are already a consumable, counted resource, not an
infinite ability.

### Door explosives

A breaching charge is a **scheduled** explosion rather than a projectile. It
detonates at a fixed slot in the round — start of Phase 1, between phases, or end
of Phase 2 — letting players sequence a breach with the movements around it. It
has a small radius but very high `wall_damage` (500): it is a tool for opening
walls and doors, not for killing. Only player-issued charges detonate (zombies
don't breach).

### Combat-via-physics consequences

Because explosions act through the atmosphere and structural fields rather than a
bespoke AoE function, several behaviors fall out for free:

- **Self-breach by overpressure** *(designed)* — a sealed room that takes enough
  blast pressure should fail its own walls, venting explosively. The mechanism
  (wall failure driven by the pressure/heat field) is the same one walls already
  use; the threshold coupling is not yet wired.
- **Vacuum disarms blasts** — a grenade near an existing breach is far less
  lethal: with the atmosphere already venting to vacuum, there is little gas to
  carry a shockwave. The pressure field produces this automatically; it is not
  special-cased.
- **Fire as the real anti-zombie weapon** — blast ignition plus the high fire
  multiplier means the lasting threat to massed zombies is the fire an explosion
  leaves behind, not the blast itself.

---

## Energy weapons (designed)

Energy weapons are the next weapon class, and their design is settled even though
they are not yet built. The key decision is **how** a beam relates to the ray
engine: a beam is *not* handled inside the lighting/heat raycaster.

A beam runs as a **serial weapon pre-phase**, once per shot, before the
deposit-only lighting pass. A single ray is marched along the shot vector and
resolves everything the beam does to the world:

1. **Skewer units.** Unlike a bullet, a beam does not stop at the first unit —
   it passes through, depositing energy damage on every unit along its path. (A
   beam should be able to line up and kill several enemies in a corridor.)
2. **Breach walls.** Where the beam crosses solid material it deposits heat and
   wall damage; a sustained beam burns through.
3. **Mutate the map + heat field.** Wall destruction and heat deposits are
   applied here, in the pre-phase, as serial edits.

Those edits then upload to the (eventual) GPU, and the lighting/heat raycaster
runs afterward on the frozen, already-updated map. This keeps the lighting kernel
simple and read-only — units stay full light-blockers for shadows, with no need
for per-channel unit attenuation inside the kernel — and puts all world-mutating
weapon logic in its own serial pass that matches the standard
*resolve-actions → update-map → simulate* loop.

The beam's **glow is just light.** It is emitted as an ordinary (transient) light
source and deposited by the normal lighting pass; the weapon ray itself carries
only damage and heat. The DDA march is a **shared primitive** with two distinct
consumers: lighting (deposit, read-only) and weapons (mutate, pre-pass). They
share the marcher and nothing else.

A reference profile from the weapon-profile design: a roughly 8° cyan beam,
~13 m range, depositing both light intensity and heat (`heat ≈ 2.0`) along its
length, with a sharp near-uniform cone and a hard cutoff. A **muzzle flash** is a
companion profile — a brief, wide, warm omnidirectional flash spawned for one or
two frames at the firing point, picked up by the lighting pass as a normal
short-lived source.

---

## Further weapon types (designed)

| Weapon | Model | Reuses |
|---|---|---|
| Flamethrower | a **directed fuel field** — a scalar gas emitted in a cone that ignites on contact with oxygen | smoke's diffusion + advection; the existing fire/oxygen combustion rule |
| Teargas | the same fuel-field pattern, with a non-combustion effect (damage/slow) instead of ignition | 100 % of the diffusion + advection pipeline |
| Shotgun | a short-range, wide-cone variant of the bullet path | the rifle's cone-march, retuned |
| Electrical arc | event-triggered bolt that seeks the nearest conductor and arcs to it | a transient muzzle-flash-style light at the origin; future water-conduction AoE |

The flamethrower and teargas are the clearest payoff of combat-via-physics:
modeling a flamethrower as a gas that flows, pools in corners, gets sucked
through breaches, and starves in vacuum requires **zero special-case weapon
code** — it is the smoke transport pipeline running on a second scalar field that
happens to burn. A flamethrower through a doorway fills the room because the fluid
solver fills the room. New weapons of this family are new fields and emission
rules, not new systems.

---

## Determinism and the facade boundary

Combat is driven from the `Simulation` step, which calls the combat module's
entry points in a fixed order each tick: scheduled door explosives at their
slots, projectile advancement and detonation, then per-unit shooting. The
combat module owns `Projectile`, `Shot`, the shooting/LoS logic, and
`apply_blast_damage`; the pure field-mutating `apply_explosion` /
`add_explosion_smoke` stay in the physics namespace and are *called into* at
detonation sites. Combat never touches a physics field directly.

Two RNG sites — the bullet cone and explosion-smoke texture — both consume the
simulation's single seeded generator, so a recorded firefight reproduces
exactly. Visual artifacts (tracers, explosion events, hit/kill events) are
emitted to caller-owned lists and event streams; they are outputs for the
renderer, never inputs to the simulation, so a headless training run produces
identical world state whether or not anyone is drawing it.

---

## Implementation status

Audited against `src/simulation/combat.py`, `src/simulation/physics.py`,
`src/simulation/simulation.py`, `src/simulation/ai_zombie.py`,
`src/simulation/unit.py`, and `config.toml`.

**Built and shipped:**

- **Rifle** — burst fire, per-bullet cone from the seeded RNG, tile-march with
  wall/unit stop, range + `has_los` gating, auto-fire in Move & Attack, zombie
  `bullet_damage_multiplier`. (`combat.py: fire_burst / process_shooting /
  auto_fire`.)
- **Line of sight** — `gmap.has_los` (Bresenham over `is_wall`) is the v1
  backing; exposed as an interface but not yet upgraded (no PVS/hierarchical, no
  heat-channel infravision implemented).
- **Melee + conversion** — zombie adjacency attack on cooldown, `killed_by_zombie`
  flag, end-of-round `convert_marines_to_zombies`. (`ai_zombie.py`.)
- **Explosions** — `apply_explosion` (wall HP + destruction, 3×3 smoothed
  `wave_source` deposit, atmosphere boost, smoke clearing, flammable ignition),
  `apply_blast_damage` (falloff + `blast_damage_threshold`), `add_explosion_smoke`.
- **Grenades** — `Projectile` with player-set fuse and straight-line travel,
  detonation in `_update_projectiles`, AP + `has_grenade` cost/refund, carry-over
  of long-fuse grenades.
- **Door explosives** — scheduled-slot detonation, player-only, high wall damage.
- **Determinism plumbing** — RNG threaded through `fire_burst` and
  `add_explosion_smoke`; event emission (`ShotFiredEvent`, `ExplosionEvent`,
  `UnitHitEvent`, `UnitKilledEvent`) wired for the renderer.

**Designed, not built:**

- **Energy weapons** — the serial weapon pre-phase, beam skewering, beam-driven
  wall breaching, heat deposit, and beam-glow-as-light source are fully specified
  but unimplemented. No energy-weapon order type, profile, or pre-phase hook
  exists in the code. (The reference `energy_weapon`/`muzzle_flash` profiles live
  only in the superseded radiation-temperature plan.)
- **Flamethrower / teargas** — the directed-fuel-field model is specified; no fuel
  field, emission rule, or combustion coupling is implemented. Depends on the
  smoke transport + fire/oxygen systems being reused.
- **Shotgun** — listed as a needed Mission 1 weapon; no code.
- **Electrical arcs** — design (recursive-midpoint visual, conductor seeking,
  water AoE hook) exists only in the superseded plan; nothing built.

**Gaps and rough edges:**

- **Heat damage to units is not implemented.** The via-physics design says units
  sample the heat buffer at their footprint and take damage in serial unit logic.
  No such sampling exists yet — current unit damage is only bullet, melee, and
  blast. Once the ray/heat engine's GameMap-owned `heat` buffer is fully in the
  sim step, this is the missing link that makes beams and fire actually hurt
  units through the field.
- **Overpressure self-breach is not wired.** Walls are destroyed by explicit
  `wall_damage`, not by the pressure field crossing a failure threshold. The
  intended emergent venting behavior does not yet happen.
- **Grenades do not bounce.** Projectiles travel in a straight line to the target
  and ignore walls in flight; throwing into a wall is not modeled as a bounce.
- **Ammunition is a stub.** `has_grenade` / `has_explosive` are integer counts on
  `Unit`; rifle ammo is unlimited. The `Inventory` class is a placeholder
  (`current_load` returns 0, no item list). A real ammo-as-inventory system is
  designed but unbuilt.
- **`apply_blast_damage` uses `getattr(u, "id", -1)`** defensively, implying unit
  IDs are not guaranteed present on every code path — a small consistency wart in
  the event-emission layer.
- **Config drift** — several explosion constants (atmosphere-boost factor 0.3,
  smoke-clear radius 0.4, ignition radius 0.7, ignition intensity 0.5) remain
  inline magic numbers in `apply_explosion` rather than living under
  `[weapons.*]` in `config.toml`.
