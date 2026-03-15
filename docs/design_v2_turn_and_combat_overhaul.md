# Design Document: Turn & Combat Overhaul (v2)

> **Status**: READY FOR IMPLEMENTATION — all major decisions resolved
> **Date**: 2026-03-15 (updated evening session)
> **Goal**: Reach the critical mass of changes needed to evaluate whether the game is fun.

---

## 1. Problem Statement

The game is not fun yet. We need enough systems working together — turn structure,
movement feel, weapons, enemies — to judge whether the core loop works. This document
defines what "enough" means and how each piece fits together.

---

## 2. Two-Phase Round Structure

Each round (triggered by pressing Space) is divided into **Phase 1** and **Phase 2**.

### Execution order within each phase (60 ticks)

Each tick resolves in this priority order:
```
1. Door explosives detonate (if scheduled for this tick — highest priority)
2. Wall/map state updates (destroyed walls removed)
3. Grenade detonations (if fuse expires this tick)
4. Shooting damage resolves (hitscan)
5. Units advance along paths (1 tile per N ticks, depending on speed)
6. Melee/contact damage (zombies adjacent to marines)
7. Death and conversion checks
8. Physics substeps (atmosphere, smoke — multiple substeps per tick as needed)
```

### The sync point

All units synchronize at the boundary between Phase 1 and Phase 2. If a unit finishes
its Phase 1 actions early, it **waits** at its destination until Phase 2 begins. This
creates a natural coordination point the player can exploit:

- Plant explosive end of Phase 1 → detonates before Phase 2 → breach with full AP in Phase 2
- One unit moves into position Phase 1 → provides covering fire Phase 2

### What the player sees during planning

The player plans the **entire round** (both phases) before pressing Space. The UI should
make the phase boundary visually clear on the timeline — a thick divider or color shift.
The player assigns actions to Phase 1 or Phase 2 for each unit.

---

## 3. Action Points and Time Points

### Action Points (AP)

- Each unit gets **2 AP per phase** (4 AP total per round).
- AP are spent on discrete actions: shoot, throw grenade, plant explosive, etc.
- Unspent AP do not carry over between phases or rounds.

### Time Points (TP)

- Movement costs **time**, not action points.
- All positions and distances are in **fine tiles** (1 tile = 1/3 meter, unit = 3×3 tiles).
- Movement speed is expressed as **ticks per tile** (how many game ticks to traverse 1
  fine tile):
  - **Move & Attack**: 9 ticks/tile (4 tiles/s, 1.33 m/s)
  - **Move with Cover**: 6 ticks/tile (6 tiles/s, 2.0 m/s)
  - **Sprint**: 4 ticks/tile (9 tiles/s, 3.0 m/s)
- Units move at a **constant speed** determined by their mode. If the destination is
  closer than the budget, they arrive early and wait at the sync point.
- Diagonal movement: **alternating 1-2 cost** (D&D 3.5 style). First diagonal step costs
  1 tile, second costs 2, repeat. Average ≈ √2. No error tracking needed.

### Simultaneous actions

A unit can **move and act** in the same phase. The time cost is whichever takes longer
(movement or action), not the sum. Example: a unit can move 4 tiles and fire twice in
one phase — the movement and shooting happen concurrently.

### Game clock

- **12 game ticks per second** (83ms per tick).
- **Phase duration**: 5 seconds = **60 ticks per phase**, 120 ticks per round.
- **Pure grid positions** — no continuous coordinates. Units are always on exact tile
  positions. Movement hops tile-to-tile at the unit's tick rate.
- Fast units (xeno sprint: 1 tick/tile) may traverse multiple tiles per tick. Check all
  tiles along the path within that tick to prevent tunneling.
- Slow units (marine attack: 9 ticks/tile) sit on a tile for multiple ticks before
  advancing to the next.
- Rendering interpolates between ticks for smooth animation.

---

## 4. Door Explosives — Detonation Timing

Door explosives (breaching charges) have **3 discrete detonation slots**:

| Slot | When | Tactical use |
|------|------|-------------|
| **Start of Phase 1** | Before anything else in Phase 1 | Full surprise breach — enemy doesn't see it coming, but you also don't know what's behind the door before committing |
| **Between Phase 1 and Phase 2** | After Phase 1 resolves, before Phase 2 begins | Safe breach — Phase 1 to position, see the result, Phase 2 to push through |
| **End of Phase 2** | After Phase 2 resolves | Preparation for next round — see what's behind the door during next planning phase |

### Key rule

**Door explosives always resolve before everything else in their timing slot.** This
means:

- Pathfinding is computed *after* the explosion — the door tile is already gone
- Units can be given move orders through the door — they won't collide with it
- Grenades thrown in the same phase can pass through the now-open doorway
- The player can trust: "if I blow it, it's gone, I can plan as if it doesn't exist"

### Planting

- Planting a breaching charge costs **1 AP**.
- The unit must be adjacent to the door (within 1 coarse tile).
- Planting can happen in any phase. The player selects the detonation slot separately.
- A charge planted in Phase 1 can detonate as early as "between Phase 1 and Phase 2".
- A charge planted in a previous round can detonate at "start of Phase 1" — this is the
  key combo: plant last phase of previous round → detonate start of next round → full AP
  for the breach.

---

## 5. Grenades

**Player-set timer** — the player sets the fuse freely (scroll wheel) from T=0 to T=10
seconds. Grenades can persist across round boundaries.

- T=0 is valid: the grenade detonates the instant it leaves the unit's hand. Suicidal,
  but sometimes the right call when overwhelmed by zombies.
- Long timers allow area denial: throw a grenade into a corridor, force enemies to move
  around it or eat the blast.
- This subsumes the "short fuse" option — a player who always wants instant detonation
  just sets T=0.3 every time.

### Grenade interaction with door explosives

Grenades thrown at the start of a phase resolve *after* door explosives in that phase.
This means you can blow a door and throw a grenade through it in the same phase. The
grenade's travel time naturally creates a small delay — it won't arrive before the
explosion clears the door.

---

## 6. Guns and Shooting

### What we need now

- Hitscan weapons (bullets travel instantly along a line).
- **High rate of fire** — shooting should feel fast and impactful.
- Shooting costs **1 AP** per fire order.
- A fire order lasts until the **end of the current phase** (not the whole round). To
  sustain fire across both phases, the player spends 1 AP in Phase 1 and 1 AP in Phase 2.
- Each unit has a primary weapon (rifle) with a large magazine.

### Shooting mechanics (initial version)

- Player assigns a **fire order** targeting a tile, direction, or enemy unit.
- During execution, the unit fires continuously toward the target for the remainder of
  the phase.
- Bullets that hit walls damage them. Bullets that hit enemies damage them.
- Accuracy: for now, simple cone of fire from unit center toward target. Bullets scatter
  within the cone. Close range = tight cone = more hits.

### Visual and audio feel

- Muzzle flash on the firing unit.
- Bullet traces (thin lines) visible briefly.
- Impact sparks on walls, blood effect on enemies.
- Sound: rapid bursts, not single shots. This is a squad breaching rooms — volume matters.
- **Rate of fire should feel aggressive** — short, punchy bursts with visible tracers.

### Future (not this patch)

- Different weapon types (shotgun, SMG, sniper, pistol).
- Accuracy modifiers based on movement mode (move & attack vs. stationary).
- Unit specialization (assault marines good at move & attack, snipers good at overwatch).
- Overwatch mechanic (reaction fire during enemy movement).

**Design hook for specialization**: The unit class should track which order type it's
currently performing. This lets us later add per-class accuracy/damage modifiers without
restructuring. Don't implement the modifiers now — just make sure the data is there.

---

## 7. Zombies

### Encounter context

The squad does **not** expect zombies in the first mission. They have only bullet
weapons. Zombies are resistant to bullets — this creates tension and forces focus fire.

### Zombie properties

| Property | Value | Notes |
|----------|-------|-------|
| HP | High (300–500?) | Resistant to bullets — takes many hits to kill |
| Speed | Slow-medium | Slower than sprinting marines, maybe similar to move & attack |
| Damage | High | One or two hits should down a marine |
| Weak to | Fire | Takes 3–5× damage from fire sources |
| Bullet resistance | High | Takes 0.2–0.3× damage from bullets |

### Zombie AI: Trigger radius system

1. Each zombie has a **trigger radius** (e.g., 8 coarse tiles). When any player unit
   enters this radius, the zombie **activates**.
2. Once activated, the zombie starts moving toward the nearest player unit.
3. **Chain activation**: Any zombie within a **propagation radius** (e.g., 5 coarse
   tiles) of an already-triggered zombie also activates. This cascades — triggering one
   zombie near a group triggers them all.
4. Zombies move and attack during the execution phase simultaneously with player units.
5. Zombies do **not** plan — they react. Their movement is computed at the start of each
   phase based on current player positions.

### Zombie conversion

When a marine is killed by a zombie, that marine becomes a zombie on the enemy team.
The converted zombie:
- Has the same HP as a normal zombie.
- **Retains the marine's visual model** (sprite/color) so the player recognizes their
  fallen teammate — visual distinction (e.g., color tint, animation change) deferred to
  later.
- Follows the same AI as other zombies.
- Does **not** retain equipment or abilities — just a plain zombie in a marine's body.

### Poison gas (deferred)

Zombies emitting poison gas on death is a cool idea but adds complexity. **Defer to a
later patch** — we have the atmosphere/smoke system ready to support it when the time
comes.

### What we need to make zombies work

1. Enemy unit spawning (place zombies on the map during level setup).
2. Trigger radius detection (distance check each phase).
3. Chain activation (BFS/flood fill from triggered zombies).
4. Simple movement AI (move toward nearest player unit, respect walls, use pathfinding).
5. Melee attack (damage on adjacent tile contact).
6. Death → check if killed by zombie → convert.
7. Health system for player units (they have HP already but no damage source exists).

---

## 8. Parameters and Tuning

All balance parameters should be externalized into a config file so we can tweak without
editing code. Use a simple format — either TOML or JSON.

### Config file structure

```
[clock]
ticks_per_second = 12              # game ticks per second (83ms per tick)
phases_per_round = 2
phase_duration_seconds = 5.0       # real-time per phase (= 60 ticks per phase)
ap_per_phase = 2

[movement]
# Speed in ticks per fine tile (lower = faster)
marine_attack_ticks_per_tile = 9   # 4 tiles/s, 1.33 m/s
marine_cover_ticks_per_tile = 6    # 6 tiles/s, 2.0 m/s
marine_sprint_ticks_per_tile = 4   # 9 tiles/s, 3.0 m/s
xeno_sprint_ticks_per_tile = 1     # 36 tiles/s, 12.0 m/s (fastest in game)

[weapons.rifle]
damage_per_bullet = 10
bullets_per_burst = 5
cone_half_angle_degrees = 3.0
range_tiles = 90                   # in fine tiles (= 30 meters)
ap_cost = 1

[weapons.grenade]
blast_radius = 6                   # fine tiles
pressure = 10.0
fuse_min_seconds = 0.0
fuse_max_seconds = 10.0
fuse_default_seconds = 0.5
ap_cost = 1
max_throw_range = 30               # fine tiles

[weapons.door_explosive]
blast_radius = 3                   # fine tiles
pressure = 5.0
wall_damage = 500
ap_cost = 1

[zombie]
hp = 400
ticks_per_tile = 9                 # same speed as marine attack move
melee_damage = 60
trigger_radius = 24                # fine tiles (= 8 meters)
propagation_radius = 15            # fine tiles (= 5 meters)
bullet_damage_multiplier = 0.25
fire_damage_multiplier = 4.0

[marine]
hp = 100
```

### Hot-reload (stretch goal)

If practical, support reloading the config file at runtime (e.g., press F5) so we can
tweak values without restarting the game. This is extremely valuable during playtesting.

---

## 9. What's NOT in this patch

To keep scope manageable, the following are explicitly deferred:

- Fire system (designed, prototyped, but not needed for zombie v1 — ironic since zombies
  are weak to fire, but we can add fire weapons in the next patch)
- Weapon specialization / accuracy modifiers per movement mode
- Overwatch mechanic
- Poison gas from dead zombies
- Multiple weapon types (shotgun, SMG, sniper)
- Vision / fog of war / line of sight
- ~~Pathfinding~~ (now included — see Section 12)
- Wave equation shockwaves (diffusion is sufficient for now)
- Unit classes with different stats
- Reinforcement learning for balance tuning (interesting idea — revisit after we have
  enough systems to meaningfully optimize)

---

## 10. Implementation Priority

Suggested order — each step should produce a testable, playable state:

1. **Config file system** — externalize all balance parameters, hot-reload with F5
2. **Pathfinding** — A* on coarse grid, temporal reservation for friendly units (Section 12)
3. **Two-phase round structure** — refactor turn system, add sync point
4. **Action points + time points** — new resource system, constant-speed movement
5. **Door explosive detonation slots** — 3 discrete timing options, resolve-before-all rule
6. **Grenade rework** — player-set timer (0–10s), cross-round persistence
7. **Guns** — hitscan shooting, sustained fire per phase, AP cost, visual feedback
8. **Zombies** — spawn, trigger radius, chain activation, A* movement, melee, conversion
9. **Tuning pass** — playtest and adjust config values until it feels right

---

## 11. Open Questions

These need answers before or during implementation:

1. ~~**Phase duration**~~: **DECIDED** — 5 seconds per phase.
2. ~~**Zombie pathfinding**~~: **DECIDED** — A* on coarse grid for zombies. Temporal A*
   for player units (see Section 12). Future AI will use neural networks.
3. ~~**Movement distances**~~: **DECIDED** — 4/5/6 coarse tiles per phase. Good for now,
   will tune via config. Enemies (not distance) are the intended movement constraint.
4. ~~**Grenade fuse**~~: **DECIDED** — Player-set timer, 0–10 seconds. Can persist across
   rounds. T=0 is suicidal but valid.
5. ~~**AP per phase**~~: **DECIDED** — 2 AP per phase, 4 per round. Fire orders last until
   end of current phase (1 AP each), so sustaining fire costs 2 AP across the round.
6. ~~**Converted zombies**~~: **DECIDED** — No equipment retained. Plain zombie with
   marine's visual model.
7. ~~**Diagonal movement**~~: **DECIDED** — Alternating 1-2 cost (D&D 3.5 style). First
   diagonal costs 1, second costs 2, repeat. Average cost 1.5 ≈ √2. Simple, no
   accumulated error tracking needed.

### New open questions

8. **Inventory explosion system**: When a unit (marine, zombie, or future enemy) dies
   near a grenade blast, any carried grenades/explosives in their inventory also detonate.
   This creates emergent chain reactions — e.g., a grenade kills a marine carrying 2
   grenades, causing a secondary explosion. Simple system, complex emergent behavior.
   **Status**: DEFERRED to future patch. Needs design thought on implementation approach
   — there are many ways to handle recursive explosions (immediate chain vs. queued vs.
   next-tick resolution, blast overlap rules, chain depth limits, etc.). Worth getting
   right rather than rushing. The system would apply uniformly to all unit types (marines,
   zombies, future enemies).

---

## 12. Pathfinding

Two distinct pathfinding systems, both operating on the **fine grid** (1 tile = 1/3 meter).
Units occupy 3×3 fine tiles — collision checks test all 9 tiles of the unit's footprint.

### 12.1 Zombie pathfinding: Standard A*

- Runs at the **start of each phase**, after door explosives have detonated.
- Goal: nearest player unit (Euclidean heuristic).
- Walls and intact doors are impassable. Destroyed doors are passable.
- Recomputed every phase — zombies always chase the current nearest target.
- Cost: uniform 1 per cardinal step, alternating 1-2 per diagonal step.
- Collision: 3×3 footprint checked at each candidate position.

### 12.2 Player unit pathfinding: Temporal A* (A* in time)

Player units use manual waypoints (click-to-move), but their paths must avoid collisions
with **friendly units**. This requires pathfinding in a 3D space: (x, y, tick).

#### The reservation table

A shared data structure tracks which fine tiles are occupied at which game ticks:

```
reservations[tick][x][y] → unit_id or None
```

- When a unit's path is computed, all 3×3 tiles it occupies at each tick are reserved.
- When computing a path for the next unit, reserved tiles are treated as impassable at
  those specific ticks.
- Order of computation matters: units planned first get priority. The UI should let the
  player see and adjust the planning order if needed (or we use a sensible default like
  selection order).
- Phase = 60 ticks. Table size: 60 × 120 × 75 ≈ 540K entries per phase. Trivial.

#### How it works

1. Player clicks a destination for a unit.
2. The game computes A* from the unit's current position to the destination, where each
   node is (x, y, tick) and neighbors include:
   - Move to adjacent tile: (x±1, y, tick+speed) or (x, y±1, tick+speed) — where speed
     is ticks_per_tile for the unit's movement mode
   - Move diagonally: (x±1, y±1, tick+speed) or (x±1, y±1, tick+2×speed) — alternating
   - Wait in place: (x, y, tick+1) — unit stays put for 1 tick
3. A tile is blocked if:
   - It's a wall/door at that tick (respecting door explosives that will have gone off)
   - Any of the unit's 3×3 footprint tiles are reserved by another friendly unit at that tick
4. The resulting path is displayed as the planned route. The unit follows it during
   execution.

#### Important constraints

- **Teammates are always aware of each other** — no fog between friendlies.
- **Enemies are NOT in the reservation table** — only friendlies avoid each other.
  Enemies and friendlies can occupy the same tile (that's combat).
- **Waiting is free** — a unit can spend ticks standing still to let a teammate pass
  through a corridor first. This should feel natural, not punishing.
- **Replanning**: If the player changes one unit's orders, all subsequent units' paths
  may need to be recomputed (their reservations shift). This should be fast enough to
  feel instant — A* on 120×75 with 60 ticks is very manageable.
- **Corridors**: Two marines (3×3 each) need a corridor at least 6 tiles wide to pass
  side by side. In narrower corridors, one waits while the other passes.
