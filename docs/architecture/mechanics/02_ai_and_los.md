# AI & Line-of-Sight

**Depends on:** [Units](01_units.md), [Grid & Coordinates](../engine/01_grid_and_coordinates.md), [Ray Engine](../engine/08_ray_engine.md)

Two systems meet in this chapter. **Line-of-sight** answers a single boolean question — can point A see point B through the world? — and is consumed by everything that needs to know whether one tile is visible from another: enemy AI deciding whether it has spotted the squad, marines deciding whether they can shoot a target, and (designed, not yet wired) the renderer deciding which enemies to draw. **AI** is the decision layer that turns world state into unit intent. Today that layer is exactly one behaviour — the zombie — but it is structured so that "which brain runs" is a property of the unit, not a branch buried in the simulation loop.

Both systems are deliberately thin. LoS is a single function behind a stable interface. AI is a per-tick pure function over the entity list and the map. Neither owns state beyond what already lives on the unit. This keeps them headless, deterministic, and trivially callable from the C++ port — they read `gmap.<field>` and unit fields, and write only unit fields.

---

## 1. Line-of-Sight

### 1.1 The interface is the contract

Every visibility query in the game goes through one method:

```python
gmap.has_los(fy1, fx1, fy2, fx2) -> bool
```

It takes two tile coordinates and returns whether an unobstructed sightline exists between them. **The signature is the load-bearing decision.** Callers state *what* they want to know (is B visible from A?) and never *how* it is answered. That separation is what lets the backing algorithm change — from the current Bresenham walk to a ray-engine query, to a precomputed visibility set — without touching a single call site.

There are exactly three callers today, and they are representative of the three roles LoS plays:

| Caller | File | Question being asked |
|---|---|---|
| Zombie trigger detection | `ai_zombie.update_zombies_tick` | Can this zombie *see* a marine, so it should wake? |
| Marine fire-order validation | `combat.process_shooting` | Can this marine *shoot* the ordered target tile? |
| Marine auto-fire targeting | `combat.auto_fire` | Which visible enemy is the nearest valid target? |

All three pass **tile-center** coordinates (`unit.center_tile_x()` / `center_tile_y()`), not the footprint corner — a unit's sightline originates from the middle of its 3×3 block, which is the visually and tactically correct origin.

### 1.2 The v1 backing: Bresenham

The current implementation walks a Bresenham line from origin to target and stops at the first wall:

```python
def has_los(self, fy1, fx1, fy2, fx2):
    # integer DDA from (fx1,fy1) toward (fx2,fy2)
    while True:
        if (x, y) == (fx2, fy2):
            return True                 # reached target unobstructed
        if self.is_wall[y, x]:
            return False                # a wall stands between A and B
        step()                          # advance one tile along the line
```

Properties worth naming:

- **Occlusion is binary.** A tile either blocks the line or it does not. The test is against the `is_wall` mask — hull, wood, and **closed doors** block; air and **open doors** do not. Glass currently blocks LoS even though the ray engine treats it as transparent for light (see §1.4 for the reconciliation).
- **It checks geometry only.** Smoke, darkness, and distance do not enter the calculation. A marine has line-of-sight to a target across a smoke-filled room at maximum range exactly as if the room were clear and the target adjacent. Range and other gates are applied *by the caller* before the LoS check, never inside it.
- **The origin and target tiles are not tested.** The walk returns `True` the instant it reaches the target tile and only tests tiles strictly between the endpoints. A unit standing on a wall tile (which cannot happen in normal play) would still "see out."
- **It is symmetric and deterministic.** Integer arithmetic, no randomness, identical on every machine — a requirement for headless training and replay.

Bresenham is the right v1 because it is correct, cheap (one integer walk per query), and its result is exactly what a per-tile occlusion grid should produce. It is explicitly *backing*, not *the design* — the design is the interface.

### 1.3 The designed backing: pairwise rays + infravision

The ray engine already owns the primitive that LoS wants: a DDA march that walks tile-by-tile across a read-only world and accumulates per-channel attenuation. The locked design re-expresses `has_los(a, b)` as a **pairwise ray** from observer to target along that same primitive, terminating at the first tile whose accumulated attenuation crosses opacity. This unifies the two notions of "can light get from here to there?" and "can this unit see there?" — they become the same march on the same world, differing only in which channel they read and what threshold they apply.

Two capabilities fall out for free once LoS rides the ray engine:

- **Per-channel occlusion.** Because the ray engine attenuates per material per channel (opaque hull = full attenuation, glass ≈ 0.1, smoke = density-driven), a sightline through glass or thin smoke can be *partial* rather than the current hard yes/no. The `has_los` boolean is the thresholded form of a continuous transmittance the march already computes.
- **Infravision is the identical query on the heat channel.** A species that sees in infrared runs the same observer→target march but tests the **heat** buffer instead of the visible-light occlusion. Cold smoke that blocks sight is transparent to it; a warm body behind that smoke is visible. No new code path — a parameter selecting which channel the query reads.

The richer machinery (hierarchical visibility, precomputed potentially-visible sets) is deferred *behind the same interface*. None of it changes a call site.

### 1.4 Units as occluders

A subtlety that distinguishes "can I walk there?" from "can I see there?": **units block sight, not just movement.** Today `has_los` tests `is_wall`, which is the static wall mask and does **not** include units — so a marine can currently see (and shoot) straight through another unit's body. The locked design closes this by stamping unit footprints into the ray engine's occlusion field each round, exactly as `stamp_units()` already stamps them into `obstacles` for the physics. The unit becomes a read-only occluder the march sees and the kernel never writes back to.

The per-channel design adds nuance the binary mask cannot express: a unit fully blocks **light** (it casts a shadow, and you cannot see a target standing behind it) but only **partially attenuates** heat/energy. This is what lets an energy beam skewer multiple bodies in a line while still casting a clean shadow — the same footprint stamp, read with a different per-channel coefficient.

---

## 2. AI

### 2.1 Where AI sits in the tick

AI runs once per tick inside the deterministic `Simulation.step`, after player actions resolve and before the world is re-stamped and stepped:

```
per tick:
  1. fire scheduled door explosives (tick 0 only)
  2. update projectiles
  3. update player movement (precomputed paths)
  4. process shooting (player fire orders + auto-fire)
  5. update_zombies_tick(gmap, units, tick)   <-- AI
  6. stamp_units(units)                        -> rebuild obstacles + occlusion
  7. physics step
  ...
  end of round: convert_marines_to_zombies(units)
```

The ordering matters. AI reads the *current* obstacle field and the *current* unit positions, moves units, and the **re-stamp on step 6** then rebuilds `obstacles` to reflect those moves before physics runs — so waves, smoke, and light see units where the AI just put them, not where they were last tick. Conversion runs at end-of-round, outside the tick loop, because it is a discrete roster change rather than a per-tick decision.

### 2.2 Brain selection is a unit property

The simulation does not contain a switch over unit types. Each unit carries `is_zombie` (source-of-truth state, mirrored by `team`), and the design routes AI by that flag: a zombie runs the zombie brain, a unit driven by the player runs no autonomous AI, and future species select a brain by their species/intelligence tier. "Zombie" is a **state, not a class** — any unit can be converted into one, keeping its base fields and inventory, and from then on the zombie brain drives it.

The forward design generalizes this into two layers that do not yet exist in code:

- **Stance drives friend/foe.** A `FactionRelationshipTable` answers `stance_between(a, b)` (Allied / Friendly / Neutral / Hostile), owned by the mission rather than the unit. This is what lets three factions fight on one map and what an AI consults to decide who is a target — replacing today's hardcoded `team == 0` / `team == 1` test.
- **Behaviour layers on top of stance.** Fight-or-flight (a species fleeing when outnumbered rather than charging) and the deferred "Gray" mind-influence attacks are *AI behaviours*, not faction properties. A unit's `nn_intelligence_tier` is designed to select which neural network drives it once learned policies replace the hand-written brain. The state encoding for that network is future work; structuring the brain as a pure function over world state keeps the door open.

### 2.3 The zombie brain

`update_zombies_tick` is the one fully-built brain. It runs three sequential passes over the unit list each tick.

**Pass 1 — Trigger detection.** Every inactive zombie checks each living marine. If the marine is within `trigger_radius` (Euclidean, tile-center distance) **and** `has_los` confirms an unobstructed sightline, the zombie activates. Radius and LoS are both required: a marine close behind a wall does not wake the zombie, and a marine in the open but beyond the radius does not either. This is the *only* place perception enters the brain.

**Pass 2 — Chain activation.** Activation spreads. A BFS-style sweep repeats `while changed`: any active zombie wakes any inactive zombie within `propagation_radius`, and the loop runs until a full pass adds nobody. **This propagation is range-only — it does not check LoS** — modelling a horde rousing each other by sound/proximity rather than sight. The sweep is correct but O(N³) worst case (the `while changed` loop over a nested N² scan); fine at current squad sizes, flagged for replacement with a proper frontier BFS when hordes grow.

**Pass 3 — Movement and combat.** Each active zombie:

1. Picks the **nearest living marine** by Euclidean distance (no LoS — once awake, a zombie knows where prey is and pursues relentlessly).
2. If within melee reach (`footprint + 1` tiles) and its attack cooldown has elapsed, it deals `melee_damage` and resets `last_melee_tick`. A marine reduced to ≤0 HP is marked dead **and flagged `killed_by_zombie`** — the flag that drives end-of-round conversion.
3. Otherwise it moves. Movement is gated by `speed_ticks_per_tile`: an accumulator increments each tick and a step is taken only when it reaches the threshold, so a slow brute moves one tile per ~10 ticks while a runner moves every ~4. On a step it follows a cached A* path, **re-pathing when the path is empty, exhausted, or every 5 steps** to track a moving target. Before each hop it re-checks passability (a wall may have been blown open or sealed since the path was computed) and drops a stale path if the next tile is now blocked.

Pathfinding is standard 8-directional A* with a 3×3 footprint passability test (`is_passable_block`). If the pathfinding module fails to import, the brain degrades to "stand still" rather than crashing. Temporal/reservation A* exists in the codebase but is deliberately not wired here — zombies are allowed to overlap.

| Parameter | Config key | Default | Role |
|---|---|---|---|
| Trigger radius | `zombie.trigger_radius` | 48 | Max distance to wake on sight |
| Propagation radius | `zombie.propagation_radius` | 15 | Chain-activation reach |
| Melee damage | `zombie.melee_damage` | 60 | Per-hit HP loss |
| Attack cooldown | `zombie.attack_cooldown_ticks` | 12 | Ticks between melee hits |
| Speed | `zombie.ticks_per_tile` | 7 | Ticks per tile moved (per variant) |

### 2.4 Conversion

`convert_marines_to_zombies` runs at end-of-round. Every team-0 unit that is dead **and** `killed_by_zombie` is flipped: `team`→1, `is_zombie`→True, revived at zombie HP, pre-activated, and renamed `Z-<name>`. **The unit keeps its inventory** — a converted marine still carrying a grenade is a future emergent hazard (a grenade cooking off in fire) with zero special-case code. The conversion flag is cleared after, so the same corpse is never converted twice.

### 2.5 Designed: AI sees what the player sees

The strongest forward idea ties AI perception directly to the systems the player experiences. Two threads:

- **Shadow-stealth.** Visibility checks factor the **light level at the target tile**, not just geometry. A marine standing in an unlit room — below the stealth threshold sampled from the light buffer — is harder for AI to detect even with a clear sightline. This turns lighting into a tactical resource: shoot out the lights, drop smoke, fight from the dark. It composes with §1.3: LoS gives geometric visibility, light level gives detectability, and the two multiply.
- **Perception = the sim grid (or the rendered image).** The longer-range idea is that an AI's *input* is the physics grid itself — or even the rendered frame — so that what is hard for the player to see is hard for the AI to see. Different species get different views of the same world: infravision reads the heat channel, a light-blind species reads only geometry, a Gray manipulates the target's relationship to reality at the perception layer. This is the natural home for the neural-network agent: a per-species observation function over `gmap.<field>` feeding a policy net selected by `nn_intelligence_tier`.

---

## Implementation status

Audited against `src/simulation/ai_zombie.py`, `src/simulation/gamemap.py`, `src/simulation/combat.py`, and `src/simulation/simulation.py`.

**Implemented (shipped, working):**

- `gmap.has_los(fy1, fx1, fy2, fx2)` — integer Bresenham walk against the `is_wall` mask. Binary, geometry-only.
- Three live LoS callers: zombie trigger detection, marine fire-order validation, marine auto-fire targeting. All pass tile-center coordinates.
- The full zombie brain: trigger detection (radius + LoS), range-only chain activation, nearest-target selection, cooldown-gated melee, speed-gated A* movement with re-pathing every 5 steps and per-hop passability re-checks, graceful "stand still" fallback when pathfinding is unavailable.
- End-of-round `convert_marines_to_zombies`, including inventory retention and the `killed_by_zombie` flag lifecycle.
- AI invoked deterministically inside `Simulation.step` (pass 5), with re-stamp afterward so physics sees moved units. Headless-clean: reads `gmap.<field>` and unit fields, writes only unit fields.
- Brain selection by unit state: `is_zombie` is the source-of-truth flag; zombie units run the zombie brain.

**Designed but not built:**

- **LoS on the ray engine.** `has_los` remains pure Bresenham; the pairwise-ray re-implementation behind the same interface is designed and locked, not coded. Per-channel partial occlusion (glass, smoke) does not affect LoS today.
- **Infravision.** Designed as the same query on the heat channel; no code, no per-species channel selector.
- **Units as sight occluders.** `has_los` tests `is_wall` only, so units do **not** block sight or fire today — a known gap. The footprint-stamp-into-occlusion mechanism is designed (and the dynamic per-channel attenuation field it would use already exists in `stamp_units` for light), but `has_los` does not consult it.
- **Renderer visibility filter.** The "don't draw enemies the player has no LoS to" integration is designed; the renderer does not yet filter by `has_los`. Fog-of-war is undecided.
- **Shadow-stealth and grid/image-as-AI-input.** Concept only; visibility checks do not factor light level, and AI input is direct entity/field access, not a per-species observation function.
- **Faction/stance layer.** `FactionRelationshipTable` / `stance_between` is designed; AI friend/foe is still the hardcoded `team == 0` / `team == 1` test. Fight-or-flight and Gray mind-influence behaviours are deferred.
- **Neural-network brains.** `nn_intelligence_tier` exists as a field on the unit/species design; no nets, no encoder.

**Gaps and rough edges:**

- Chain activation is O(N³) worst case (`while changed` over a nested N² scan). Correct for current squad sizes; needs a frontier BFS before large hordes.
- Units are invisible to LoS — marines can see and shoot through each other and through enemy bodies. Closing this requires `has_los` to consult an occlusion field that includes stamped unit footprints.
- LoS ignores smoke and darkness entirely. Tactically, smoke does not block sight today; it only affects the renderer.
- Zombie pursuit uses no LoS after activation and never loses a target — relentless by design, but offers no "break line of sight to escape" counterplay until the ray-engine LoS and stealth layers land.
- Temporal/reservation A* is unused for zombies (overlap permitted), consistent with the player path system.
