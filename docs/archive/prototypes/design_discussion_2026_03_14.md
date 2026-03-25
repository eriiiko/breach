# Design Discussion — 2026-03-14

Session focus: wind, fire-wind interaction, computation cost, resolution, simultaneous turns, path to playable demo.

---

## Topic 1: Wind

### Current State
- **No explicit wind field.** Wind is derived as the negative gradient of the atmosphere (pressure) field: **v** = −∇p.
- Atmosphere evolves via diffusion: ∂p/∂t = D∇²p. No wave properties — that's only for explosions.
- Smoke advection uses the atmosphere gradient as wind direction (implemented in `smoke_sim.py:167-175`). The advection equation: ∂smoke/∂t = +∇p · ∇smoke.
- Smoke also has its own slow diffusion (spreading even without wind).

### What Works
- Smoke gets pulled toward hull breaches (gradient points toward low pressure).
- Smoke fills sealed rooms densely (no gradient → no advection, only diffusion).
- Opening a door creates a transient pressure gradient → brief wind through doorway.
- Wind arrows visible in `wind_test.gif` (3-panel prototype with quiver plot).

### Wind as a Game Mechanic
Wind arises naturally from any pressure differential: hull breaches, fans (constant pressure sources), explosions. No special-case code needed — all sources just modify the atmosphere field, and wind follows from the gradient.

### Explosions Create Wind
Real explosions create massive blast wind (primary killer at medium range). Current implementation has explosions (wave eq) and atmosphere (diffusion) completely disconnected.

**Decision:** Explosions deposit overpressure into the atmosphere field at the detonation site:
- Clear smoke on 3×3 tiles around impact (vaporized by the blast).
- Spike atmosphere pressure at detonation tile (overpressure → gradient → wind).
- Smoke farther away gets pushed by the resulting wind.
- No coupling between PDE systems needed — simple game-logic rule.

### Validated in Prototype
`wind_test.py` / `wind_test_small.py`: empty room, hull breach on left wall, three smoke patches, mid-simulation explosion spike. Shows atmosphere venting, smoke advection, and explosion wind interaction.

---

## Topic 2: Atmosphere as Wave Medium

### Problem Identified
The wave equation currently propagates shockwaves even through vacuum (atmosphere = 0). This is physically wrong — no medium means no wave. Breaks the design principle of emergent behavior from correct physics.

The design doc *claims* "grenade near hull breach = harmless" as emergent behavior, but the current implementation achieves this via vacuum tiles as absorbers, not from the actual absence of medium.

### Decision: Scale Wave Propagation by Local Atmosphere

Use the atmosphere field to modulate wave propagation. One extra multiplication per iteration:

```python
# Before (propagates through vacuum):
p_next = 2 * p_now - p_prev + r * lap

# After (atmosphere is the medium):
p_next = 2 * p_now - p_prev + r * atmosphere * lap
```

- Vacuum (atmosphere = 0): wave cannot propagate. Grenade in space is silent.
- Half-depressurized room (atmosphere = 0.5): wave propagates with reduced strength.
- Normal pressure (atmosphere = 1.0): full propagation.

### Emergent Consequences (Free)
- Grenade near hull breach: energy vents through breach into vacuum — genuinely harmless, not just boundary-absorbed.
- Partially depressurized room: explosion is weaker — tactical implication for breaching compartments.
- Sealed room at full pressure: maximum blast (reflections compound, no energy escape).
- Mid-battle decompression changes how subsequent explosions behave.

### Full Explosion Chain
1. Grenade detonates → wave equation runs, modulated by atmosphere field
2. Wave propagates through available atmosphere → peak gradient = damage
3. Overpressure deposited into atmosphere field at detonation site
4. Atmosphere gradient = wind → pushes smoke, fire, entities
5. Smoke cleared at impact site (vaporized)

This keeps the two PDE systems (wave, diffusion) cleanly separated. The connection is a simple game-logic rule after the wave resolves, not a mathematical coupling.

### Implementation Cost
Negligible — one element-wise multiply per wave equation iteration. The atmosphere array is already in memory.

---

## Topic 3: Fire-Wind Interaction

### Current Fire System
Fire spreads by checking neighbors in a 2-tile radius (`smoke_sim.py:196-201`). Isotropic — no directional bias. Fire dies when local atmosphere drops below 0.60.

### Temperature as a Scalar Field
Add a temperature field on the same grid. Same Laplacian infrastructure:
- **Diffusion** (heat conduction through air): cheap, reuse same equation. Physically slow in air — not the main fire-spreading mechanism, but essentially free to include.
- **Advection** (wind carries heat): same mechanism as smoke advection. Wind blows hot air downwind.

Tiles ignite when: `temperature > material.ignition_temp AND atmosphere > O2_threshold AND material.flammable`.

### Radiation vs Diffusion vs Wind
The dominant heat transfer mechanism for fire spreading is **radiation** (infrared), not conduction through air. Radiation is line-of-sight, falls off ~1/r², blocked by walls. This is essentially the same raycast needed for the lighting system — fire IS a light source, and its heat radiation follows the same paths as visible light. When the 2D raycasting light system is built, fire radiation comes almost free from it.

**Decision:** Fire radiation shares the lighting raycast system. Raycast once per turn (or when fire/walls change) to produce a "heat exposure" map. Between raycasts, use intensity + material properties to infer turns-to-ignition — no continuous computation needed. Recalculate when: fire spreads to new tiles, walls are destroyed (new LOS paths), or fire extinguishes. Event-driven, not per-substep.

For now (pre-lighting system), wind-biased spreading is the practical substitute.

### Wind-Biased Fire Spreading (Primary Mechanism)
Use the wind vector at each burning tile to bias fire spread direction:

```python
wind_x, wind_y = -grad_atm_x, -grad_atm_y

# For each neighbor of a burning tile:
dot = neighbor_dir_x * wind_x + neighbor_dir_y * wind_y  # positive = downwind
spread_modifier = 1.0 + WIND_FIRE_FACTOR * dot
```

Downwind neighbors catch fire faster, upwind neighbors barely catch at all. Matches real wildfire behavior.

### O2 Delivery: Wind >> Compression
Back-of-envelope calculation: wind delivers orders of magnitude more O2 than shockwave compression.
- Wind at 5 m/s: ~1 m³/s of O2, continuously
- Shockwave at 5 atm: 5× density for ~1ms = equivalent to 0.005 seconds of normal air

**Conclusion:** Shockwave itself is irrelevant for feeding fire. The wind created by the explosion's pressure deposit is what feeds fire. Full chain:

Explosion → shockwave (damage only) → pressure deposited in atmosphere → wind → wind feeds fire

No compression-ignition system needed.

### Shockwave Ignition
Shockwaves CAN ignite via adiabatic compression heating (fuel-air explosive principle), but for typical game explosions (grenades, etc.) this is marginal. Decision: shockwaves can ignite tiles that are already hot (near existing fire, above a pre-ignition threshold) but not cold tiles. Physically reasonable, tactically interesting.

### Material Properties (Extended)

```python
MATERIAL_PROPERTIES = {
    Material.WOOD:  {"ignition_temp": 300, "fuel": 60,  "burn_rate": 1.0, "heat_emission": 5.0},
    Material.FUEL:  {"ignition_temp": 150, "fuel": 40,  "burn_rate": 3.0, "heat_emission": 15.0},
    Material.STEEL: {"ignition_temp": 999, "fuel": 0,   "burn_rate": 0,   "heat_emission": 0},
}
```

- **ignition_temp**: temperature threshold for catching fire
- **fuel**: how long the tile burns (consumed while burning)
- **burn_rate**: how fast fuel is consumed
- **heat_emission**: how much heat the fire produces per step

### Combined Ignition Formula

All three fields are already computed for other systems. Combine per-tile, once per turn:

```python
heat_exposure    # from raycast (radiation from nearby fires)
wind_dot         # dot(wind_direction, direction_from_nearest_fire) — downwind bias
atmosphere       # O2 availability

ignition_pressure = (
    heat_exposure * HEAT_WEIGHT
    + wind_dot * WIND_WEIGHT
    + atmosphere * O2_WEIGHT
)

tile.ignition_accumulator += ignition_pressure
if tile.ignition_accumulator >= material.ignition_temp:
    ignite(tile)
```

One multiply-add per tile per turn. The expensive parts (raycast, gradient) are already done for lighting and atmosphere.

**Properties:**
- Radiation only (no wind, normal atm): fire spreads slowly in all visible directions
- Radiation + wind: fire races downwind, barely spreads upwind
- Low atmosphere: even with heat and wind, fire can't ignite — decompression as fire suppression tactic
- High wind + no direct radiation: wind carries heat around corners via temperature advection (if enabled)
- Accumulator means fire doesn't ignite instantly — moderate heat over several turns eventually catches. Feels realistic.
- Weights are tuning knobs for gameplay feel.

### Implementation Priority (Layered)

| Layer | Mechanism | Cost | Priority |
|-------|-----------|------|----------|
| 1 | Contact spreading (current) | Done | ✅ |
| 2 | Wind-biased spreading (dot product) | Very cheap | High |
| 3 | Temperature field + diffusion | Cheap (reuse Laplacian) | Medium |
| 4 | Radiation via raycast (shares lighting) | Medium | Later |

Each layer adds realism independently. Layers 1+2 are sufficient for a playable game.

---

## Topic 4: Wave Equation Computation Cost

### Core Question
Can the wave equation run every turn, always, without optimization?

### Grid Dimensions
- Fine tile: 1/3 m (a marine occupies 3×3 fine tiles = 1m × 1m)
- Ship example: 50m × 30m = 150×90 fine tiles = 13,500 tiles

### CFL Constraint
dt ≤ dx / (c·√2). With dx = 1/3 m, c = 343 m/s: **max dt ≈ 0.69 ms**.

### Performance (Fine Grid, C++)

| Metric | Value |
|--------|-------|
| Tiles | 13,500 |
| Iterations (~3 traversals) | ~1,300 |
| C++ per iteration | ~3 µs |
| **Total for full explosion** | **2-5 ms** |
| Physics dt | 0.69 ms |
| **Ratio: compute vs real time** | **230× faster than reality** |

### Are Boundary Conditions Extra Expensive?
No. The Neumann BC is implemented as `np.where(wall, p_center, p_neighbor)` — same cost everywhere. Every tile gets identical operations regardless of walls. Maze vs open room: same computation cost.

### Decision: Fine Grid, No Coarse Optimization Needed
C++ computes the wave equation 230× faster than real time on the fine grid. A full explosion (all iterations) completes in 2-5ms — well within a 16ms frame budget. Multiple simultaneous explosions fit comfortably.

Coarse-grid optimization (3×3 fine → 1 coarse tile) was explored: ~27× speedup, sub-0.1ms per explosion. Kept as a back-pocket option for very large ships or weak hardware, but not needed for the target case.

### Coarse Grid Notes (If Revisited Later)
- 3×3 coarse tiles: dx=1m, dt≈2ms, ~0.05ms per explosion
- Wall/air density ratio per coarse tile → partial propagation
- Corridor alignment: snap level design to coarse grid, or use density fallback for mid-battle destruction
- Damage mapped back to fine-grid walls within the 3×3 block

---

## Topic 5: Dynamic / Variable Resolution

**Skipped.** Original idea was to use different resolutions for different physics layers or reduce resolution dynamically. Unnecessary now — fine grid (1/3 m tiles) is fast enough for all systems including the wave equation. Compute everything at full resolution, always.

---

## Topic 6: Simultaneous Turns

### Reference Game
**Frozen Synapse** (2011, Mode 7 Games) — remarkably similar concept. Simultaneous turn-based tactics, timeline planning, real-time execution playback. Key difference: no environmental physics. Worth playing for UX/planning interface inspiration.

### Turn Structure

Each turn has **T timesteps** (T=5 or 6, tunable after playtesting). Squad size: 2-6 units.

```
PLANNING PHASE (time paused, physics frozen)
├── Select Unit 1 → place orders on timeline T=1..T
├── Select Unit 2 → place orders on timeline T=1..T
├── ...
├── Select Unit N → place orders on timeline T=1..T
├── Can go back, cancel (Escape = clear all), redo any unit's orders
├── UI: movement lines drawn on map, markers show unit position at each timestep
└── Hit "End Turn"

EXECUTION PHASE (all players' orders resolved simultaneously)
├── T=1: all actions execute, physics advance
├── T=2: all actions execute, physics advance
├── ...
├── T=T: final actions, physics settle
└── Plays out in real time (many render frames per T), like a replay seen for first time
```

### Actions and Timing
- **Movement**: consumes timesteps (1 tile per timestep while moving)
- **Free actions**: throw grenade, shoot — can overlap with movement at the same T
- **Duration actions**: pick up weapon, breach door — consume timesteps
- **Grenade detonation**: player-chosen, T_throw + 1..10. Wave equation runs at detonation T.
- **Stance orders**: move under cover, sprint, move and attack, overwatch/aim

### Simultaneous Execution
All players plan independently (can't see enemy orders). All orders execute at the same time. This creates the core tension: you plan into the unknown. In real life you can't make decisions continuously — when you take the leap, you're already in the air.

### Physics During Execution
Physics runs live during execution — each timestep advances atmosphere, fire, smoke, wind. A grenade detonating at T=3 creates smoke/wind that affects visibility and fire at T=4. Persistent fields (atmosphere, smoke, fire, temperature) carry over between turns naturally — they don't care about turn boundaries.

### Fog of War
- **Planning phase**: units see only what they can see right now (start-of-turn state)
- **Execution phase**: fog of war updates live as units move and gain/lose line of sight

### Conflict Resolution (Unit Collisions)
Depends on unit states and relationship:

**Friendly units:**
- Basic pathfinding-in-time avoidance (units aware of each other's planned paths)
- Cooperation quality varies by unit pairing (trained together = better coordination)

**Enemy encounters (determined by both units' states):**
- Sprint vs Sprint → collision, both fall/stunned
- Sprint vs Move+Attack → sprinter at disadvantage, possibly taken down
- Move+Attack vs Move+Attack → melee engagement, stats determine outcome
- These are emergent from the state combinations, not special-cased

### Overwatch / Aim Orders
- "Expect enemy in direction D at timestep T_target" → unit aims and prepares
- Boosted reaction time and accuracy in that cone at T_target
- Reduced effectiveness at other timesteps (or no drawback — TBD after playtesting)
- Defaults to T=0 when enemies are already visible
- When a unit in overwatch sees an enemy: shoots ~instantly (modified by reaction stat)

### Death Mid-Turn
If a unit dies at T=2, its orders for T=3+ are cancelled. **Other units' orders continue blindly** — this is intentional. You planned based on assumptions; if those assumptions break (ally dies, enemy not where expected), your plan plays out anyway. This is the cost of commitment and the reward of good planning.

### Open for Playtesting
- Exact T per turn (5-6 range)
- Squad size sweet spot (2-6)
- Overwatch drawback vs free aim bonus
- Melee resolution details
- Reaction time stats and modifiers
- Whether planning phase is timed (adds pressure) or untimed (pure strategy)

---

## Topic 7: Path to Playable Demo

### Engine Decision: Pygame Prototype → Production Engine Later

**Rationale:** Code-only tools (Pygame, Raylib) work best for AI-assisted development — everything is text, no GUI/editor workflow. Physics prototypes already exist in Python + NumPy. Pygame provides: window, keyboard/mouse input, drawing, text rendering. That's all that's needed to test gameplay mechanics.

Production engine choice (Godot, Raylib C++, etc.) deferred until gameplay is proven in the prototype.

### Minimal Playable Demo — Build Order

1. **Pygame renderer** for existing grid + physics (walls, atmosphere, smoke, fire as colored tiles)
2. **Unit representation** — clickable marines on the grid, selection, info display
3. **Order placement UI** — click to set waypoints, timeline bar (T=1..5), action assignment
4. **Execution phase** — orders play out in real time, physics advance per timestep
5. **Basic enemy AI** — one enemy unit, simple behavior (move toward player, shoot on sight)
6. **→ Playable loop**: plan orders, execute, see results, next turn

### What This Tests
- Does simultaneous turn planning feel fun?
- Is T=5 the right number of timesteps?
- How does environmental physics interact with tactical decisions?
- What squad size feels right?
- What actions/orders are most used?

### Reference
Frozen Synapse (2011) — closest existing game to this design. Worth playing for planning UI inspiration (timeline scrubbing, movement lines, predicted outcomes).
