# Breach — Architecture Document

*Single source of truth for the game's architecture. Describes the current Python prototype (as-implemented), the target C++ simulation, and all planned systems. Intended to guide both incremental Python development and the C++ port.*

*Last updated: 2026-03-19*

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [System Overview](#2-system-overview)
3. [Grid & Coordinate System](#3-grid--coordinate-system)
4. [Material System](#4-material-system)
5. [GameMap — World State](#5-gamemap--world-state)
6. [Physics Systems](#6-physics-systems)
   - 6.1 Shared Laplacian
   - 6.2 Wave Equation (Explosions)
   - 6.3 Atmosphere Diffusion (Decompression)
   - 6.4 Smoke Dynamics (Diffusion + Advection)
   - 6.5 Fire Simulation
   - 6.6 Temperature & Heat Conduction (NEW)
   - 6.7 Wind/Fire Interaction
   - 6.8 Fluid Simulation (NEW — under development)
7. [2D Raycasting System (NEW)](#7-2d-raycasting-system)
8. [Lightning Bolt Effect (NEW)](#8-lightning-bolt-effect)
9. [Units & AI](#9-units--ai)
10. [Turn System & Execution](#10-turn-system--execution)
11. [Combat](#11-combat)
12. [Pathfinding](#12-pathfinding)
13. [Rendering](#13-rendering)
14. [Configuration & Hot-Reload](#14-configuration--hot-reload)
15. [C++ Port Strategy](#15-c-port-strategy)
16. [Known Issues & Refactoring Targets](#16-known-issues--refactoring-targets)
17. [ML & Neural Network Considerations](#17-ml--neural-network-considerations)
18. [Design Document Roadmap](#18-design-document-roadmap)

---

## 1. Design Philosophy

These principles govern all design decisions. They are non-negotiable.

**Systems, not scripts.** Every mechanic is a system that interacts with other systems through shared fields. No special-case code for specific gameplay scenarios. If you're writing an if-statement for a specific scenario, you're doing it wrong.

**Emergent complexity from simple rules.** Explosion breaks hull -> atmosphere vents -> smoke gets sucked out -> fire starves near breach. This chain requires zero scripting — it falls out of three systems reading and writing shared fields.

**Full physical simulation.** When something is happening, every in-game second is fully simulated. No shortcuts like "simulate only a fraction of the game time to save compute." However, skipping simulation when genuinely nothing is happening is fine — that's not cheating, that's being smart.

**Game time = real time.** One in-game second equals one real-world second during execution. Slow-motion is purely aesthetic (the simulation still runs fully, just displayed slower).

**Data-driven parameters.** All tunable values live in configuration files (config.toml), not hardcoded in source. This enables hot-reload during development and iteration without recompilation.

**Prototype in Python, ship in C++.** Python (with numpy) is for rapid iteration and visual debugging. C++ is for performance. The architecture must work in both. The transition is incremental: one system at a time, sharing memory through pybind11.

**Neural-network-compatible.** No architectural decision should foreclose the option of training a neural network agent. Grid-based state maps directly to CNN input (stacked feature planes, cf. AlphaStar). Clean serialization and headless simulation serve this goal.

---

## 2. System Overview

### Two-Layer Architecture

The architecture is split into two strictly separated layers:

**Simulation Layer** — A self-contained simulation that owns all game rules and state. This layer has **no dependency on any game engine** and can run headless for AI training and automated testing. It exposes a clean interface:
- `get_state()` — returns current world state (tile objects + cached arrays)
- `apply_action(action)` — applies a player or AI action to the simulation
- `step()` — advances the simulation by one tick

**Presentation Layer** — The game engine (pygame prototype, Raylib target) reads simulation state each frame and renders it. Player input is captured by the engine, translated into game actions, and forwarded to the simulation. The presentation layer **never modifies game state directly**.

This separation enables:
- Headless self-play for neural network training at scale (no rendering overhead)
- Parallel development: simulation logic progresses independently of visual implementation
- Automated testing without a rendering engine

### System Diagram

```
                     +------------------+
                     |    Config (TOML) |
                     +--------+---------+
                              |
                     +--------v---------+
                     |     GameMap       |  World state: tile objects + cached arrays
                     |  (numpy / C++)   |  Shared memory between all systems
                     +--------+---------+
                              |
          +-------------------+-------------------+
          |           |           |           |    |
     +----v----+ +---v---+ +----v---+ +-----v-+ +v--------+
     | Wave Eq | | Atmo  | | Smoke  | | Fire  | | Temp    |
     | (blast) | | (diff)| | (adv)  | | (burn)| | (cond)  |
     +---------+ +-------+ +--------+ +-------+ +---------+
          |           |           |           |        |
          +-------------------+-------------------+----+
                              |
                     +--------v---------+
                     |    Raycaster     |  Light, heat radiation, energy weapons
                     +--------+---------+
                              |
                     +--------v---------+
                     |    Renderer      |  Pygame (prototype) / Raylib (target)
                     +------------------+
```

All physics systems read and write to the same GameMap fields. The raycaster reads geometry + smoke, writes to light_map and heat deposits. The renderer reads everything, writes nothing.

---

## 3. Grid & Coordinate System

| Property | Value |
|---|---|
| Tile size | 1/3 meter (0.333 m) |
| Unit footprint | 3x3 tiles (1 m x 1 m) |
| Coordinate origin | Top-left (0,0), x increases right, y increases down |
| Grid dimensions | Level-dependent (test map: 120 x 75 tiles = 40 m x 25 m) |

Grid dimensions are set per level. The test map uses `map_w=40, map_h=25` (in unit-lengths), giving 120 x 75 tiles.

**Unit positioning:** A unit's position `(fx, fy)` is the top-left corner of its 3x3 tile block, always an integer tile coordinate. **Units are always on exact tile positions** — there are no continuous/float coordinates in game state. Movement hops tile-to-tile at the unit's tick rate (e.g. one tile every 9 ticks for Move & Attack). The renderer interpolates between the previous and current tile for smooth visual animation, but the game state is purely discrete.

*(See `docs/design_v2_turn_and_combat_overhaul.md` Section 3 for the full movement design.)*

---

## 4. Material System

Materials are data-driven. Adding a new material = one row in the table. All properties derive from the material ID.

### Material IDs

| ID | Name | In code |
|---|---|---|
| 0 | Air | `MAT_AIR` |
| 1 | Hull (metal) | `MAT_HULL` |
| 2 | Wood | `MAT_WOOD` |
| 3 | Door | `MAT_DOOR` |
| 4 | Steel | `MAT_STEEL` |
| 5 | Glass | `MAT_GLASS` |

### Property Table (config.toml)

Current config format: `[materials]` section, arrays of `[hp, reflectivity, absorption, flammable, passable]`.

**Target format** (extended for new systems):

```toml
[materials.air]
hp = 0
reflectivity = 0.0
absorption = 0.0
flammable = false
passable = true
conductivity = 0.0       # thermal: no conduction through air
ignition_temp = 0.0       # not flammable
light_blocks = false

[materials.hull]
hp = 300
reflectivity = 0.9
absorption = 0.1
flammable = false
passable = false
conductivity = 50.0       # metal: heat spreads fast along hull
ignition_temp = 0.0       # doesn't burn
light_blocks = true

[materials.wood]
hp = 60
reflectivity = 0.4
absorption = 0.5
flammable = true
passable = false
conductivity = 0.15       # slow heat conduction
ignition_temp = 300.0     # catches fire at this temperature
light_blocks = true

[materials.door]
hp = 40
reflectivity = 0.3
absorption = 0.3
flammable = true
passable = true
conductivity = 0.3
ignition_temp = 280.0
light_blocks = true

[materials.steel]
hp = 200
reflectivity = 0.8
absorption = 0.1
flammable = false
passable = false
conductivity = 45.0       # metal, slightly less than hull
ignition_temp = 0.0
light_blocks = true
blast_resist = 0.8        # high blast resistance

[materials.glass]
hp = 15
reflectivity = 0.1
absorption = 0.05
flammable = false
passable = false
conductivity = 1.0        # moderate
ignition_temp = 0.0
light_blocks = false       # transparent — light passes through, smoke blocks do not
blast_resist = 0.0         # shatters easily
```

### Cached Arrays (derived from material grid)

Rebuilt when tiles change (wall destroyed, door opened, etc.) via `_update_caches()`:

- `is_wall` — True where material is hull, wood, or closed door
- `is_vacuum` — True at map edges (hull boundary tiles destroyed = hull breach)
- `flammable` — True where material is wood or door
- `obstacles` — `is_wall` + living unit positions (updated each tick via `stamp_units()`)

---

## 5. GameMap — World State

### Tile Objects + Cached Arrays (hybrid)

The map is a 2D grid of **Tile objects** — each tile stores its material, hit points, fire state, liquid contents, and temperature. This object-based representation is the **authoritative state**: easy to extend (adding a property = adding a field), naturally handles mixed state (type information alongside continuous values), and enables clean serialization.

For physics and propagation, the simulation maintains **cached arrays** derived from tile state: pressure, smoke density, light-blocking, gas-blocking, walkability, flammability. These caches are rebuilt on tile change (`on_tile_changed(x, y)`) and allow all propagation systems to operate as fast bulk array operations (NumPy in prototype, raw arrays in C++).

This hybrid gives the best of both worlds: rich per-tile data for game logic, and fast array operations for physics.

**Python prototype note:** The current prototype uses pure numpy arrays as primary storage (no tile objects yet). This works for iteration. The tile object layer is the target architecture for C++ and will be introduced during the port.

**Serializable state:** Because all game state lives in tile objects + entity list, saving the game = dumping those structures to disk. This enables save/load, replay (initial state + action sequence), and undo (snapshot before action, restore on undo) essentially for free.

### Fields (cached arrays)

| Field | Type | Range | Description |
|---|---|---|---|
| `material` | int8 | 0-5 | Material ID per tile |
| `wall_hp` | float32 | 0-300 | Current HP (0 = destroyed) |
| `atmosphere` | float32 | 0-20 | Air pressure (1.0 = normal, 0.0 = vacuum) |
| `wave_p` | float32 | unbounded | Pressure wave deviation (explosion shockwave) |
| `wave_v` | float32 | unbounded | Wave velocity (dp/dt) |
| `wave_source` | float32 | 0+ | Pressure source (fed into wave_p over time) |
| `smoke` | float32 | 0-1 | Smoke density (0 = clear, 1 = opaque) |
| `fire` | float32 | 0-1 | Fire intensity on flammable walls |
| `temperature` | float32 | ambient+ | **NEW** — temperature on solid tiles (Kelvin or arbitrary) |
| `light_map` | float32 | 0+ | **NEW** — accumulated light intensity per tile |
| `liquid_type` | int8 | 0-N | **PLANNED** — none, blood, water, fuel (see Liquids below) |
| `liquid_depth` | float32 | 0-1 | **PLANNED** — volume of liquid in tile |
| `obstacles` | bool | 0/1 | Walls + unit positions (combined each tick) |
| `is_wall` | bool | 0/1 | Static wall mask |
| `is_vacuum` | bool | 0/1 | Vacuum boundary mask |
| `flammable` | bool | 0/1 | Can catch fire |

### Key methods

- `stamp_units(units)` — rebuild `obstacles` = `is_wall` + living unit positions. Called every tick. Units are treated as walls by all physics (waves reflect, smoke blocked, etc.). When tiles transition from obstacle to free (unit moved or died), atmosphere is filled with the mean of passable neighbors — avoiding artificial vacuum pulses.
- `destroy_wall(fy, fx)` — set material to air, update HP/caches, handle hull breach (edge tile becomes vacuum). Interior walls are filled with neighbor-mean atmosphere rather than hardcoded values. This preserves pressure differentials (wall between high/low pressure rooms still creates equalization rush) while avoiding artificial vacuum spikes when equal-pressure walls break.
- `is_passable_block(fy, fx)` — check if a 3x3 unit block can occupy this position.
- `has_los(fy1, fx1, fy2, fx2)` — Bresenham line-of-sight check against `is_wall`.

---

## 6. Physics Systems

### 6.1 Shared Laplacian

All diffusion-based systems use the same discrete Laplacian with Neumann boundary conditions:

```python
def compute_laplacian(p, wall):
    """4-neighbor Laplacian. Where neighbor is wall, mirror center value (Neumann BC)."""
    up    = roll(p, +1, axis=0);  up    = where(roll(wall, +1, axis=0), p, up)
    down  = roll(p, -1, axis=0);  down  = where(roll(wall, -1, axis=0), p, down)
    left  = roll(p, +1, axis=1);  left  = where(roll(wall, +1, axis=1), p, left)
    right = roll(p, -1, axis=1);  right = where(roll(wall, -1, axis=1), p, right)
    return up + down + left + right - 4.0 * p
```

This gives: reflection off walls, diffraction through doorways, channeling through corridors. The `wall` parameter is `gmap.obstacles` (walls + units), so units block waves and gas flow.

### 6.2 Unified Atmosphere Solver (IMEX: Explicit Wave + Implicit Diffusion)

**Design document:** See `docs/atmosphere_solver_analysis_and_patch_plan_20260319.md` for full Fourier stability analysis.

**Architecture:** Two-field system — `wave_p` (zero-mean acoustic shockwave) + `atmosphere` (bulk air pressure). Single `AtmosphereSolver` class with a single-step API (`step(dt)`). Python orchestrates the substep loop.

**Physics:** Per substep:

```
Step 1 — Explicit wave kick (on wave_p):
  v += dt * (c² * Δ(wave_p) - γ * v)
  wave_p += dt * v

Step 2 — Transfer wave anomaly → atmosphere:
  atmosphere += (wave_p - mean(wave_p)) * transfer * dt

Step 3 — Implicit diffusion on atmosphere (Gauss-Seidel):
  Solve (I - D*dt*Δ) atm_new = atm_current
  via red-black Gauss-Seidel iteration (8 iterations default)
```

The implicit diffusion factor `1/(1+μσ)` is always positive and ≤1 for any μ — **unconditionally stable**. No diffusion CFL limit. Only the wave CFL matters.

**Source feeding:** Explosions deposit into both `wave_source` (for shockwave propagation) and directly into `atmosphere` (for sustained pressure gradient that drives smoke transport). The wave_source deposit is smoothed over a 3×3 kernel (`[1,2,1; 2,4,2; 1,2,1]/16`). Each substep feeds a rate-limited fraction from wave_source into `wave_p` (`max_source_per_step` cap). The direct atmosphere deposit is safe because IMEX implicit diffusion handles spikes.

**Parameters:**

| Parameter | Config key | Default | Description |
|---|---|---|---|
| `c` | `physics.wave_c` | 300.0 | Wave speed (tiles/s) |
| `damping` | `physics.wave_damping` | 3.0 | Wave velocity damping (1/s) |
| `transfer` | `physics.wave_transfer` | 0.5 | wave_p → atmosphere transfer rate (1/s) |
| `feed_rate` | `physics.source_feed_rate` | 200.0 | Source → wave_p feed rate (1/s) |
| `D_atm` | `physics.d_atm` | 200.0 | Atmosphere diffusion coefficient (IMEX = no CFL limit) |
| `breach_rate` | `physics.breach_rate` | 5.0 | Relaxation rate at vacuum (1/s) |
| `max_source_per_step` | `physics.max_source_per_step` | 10.0 | Max wave energy fed per substep |
| `gs_iters` | — | 8 | Gauss-Seidel iterations per substep |

**CFL:** `dt = 0.5 / c ≈ 1.67 ms`. ~50 substeps per tick.

**Boundary conditions:**
- **Neumann BC** on all Laplacians: obstacles reflect (neighbor → self value). Border vacuum is also `obstacles` → reflected. Breach vacuum is NOT obstacles → waves/diffusion propagate into it.
- **Vacuum relaxation**: `atmosphere *= (1 - η)` where `η = breach_rate * dt`. Smooth drain, not hard Dirichlet.
- **2-tile sponge layer**: Only seeds from **exposed vacuum** (breach tiles that are NOT obstacles). Border vacuum blocked — sponge doesn't reach through sealed hull. Inner sponge: strong wave_v damping + atmosphere relaxation + wave_source zeroing. Outer sponge: moderate damping.
- **Neighbor-mean fill**: When walls are destroyed or units vacate, atmosphere is filled with the mean of passable neighbors. No artificial vacuum pulses.

**Wind:** Computed as gradient of total pressure (`atmosphere + wave_p`) every substep, output alongside the atmosphere fields.

**Gameplay:** Interior starts at 1.0 atm. Sealed ship holds pressure (sponge blocked by hull). Hull breach → relaxation drains, diffusion flows outward, sponge damps ringing. Sealed compartments hold pressure. Suffocation below 0.3 atm.

### 6.3 Atmosphere Diffusion

Diffusion is now handled inside the unified atmosphere solver (§6.2) via implicit Gauss-Seidel. No separate diffusion pass.

### 6.4 Smoke Dynamics (Diffusion + Advection)

**Physics:** Wind-dependent diffusion + advection by precomputed wind field.

```
D_effective = d_smoke * (1 + wind_diffusion_scale * |wind|²)       # turbulent mixing
actual_dt = dt * dt_scale                                          # time amplification
smoke += D_effective * actual_dt * laplacian(smoke)                # diffusion
smoke -= advection_rate * actual_dt * (wind . grad_smoke)          # wind transport
```

**Advection:** Uses the `grad_p · grad_smoke` formulation (matching prototype `wind_test.py`). Wind is `−grad(atmosphere + wave_p)`, precomputed by the AtmosphereSolver every substep — no redundant gradient computation.

**Wind-dependent diffusion:** Diffusion coefficient scales with local wind magnitude **squared** (`wx² + wy²`). The quadratic response means calm areas are barely affected (smoke holds shape) while high-wind areas (breaches, explosions) get strong diffusion. This prevents checkerboard oscillations near high-wind boundaries while keeping interesting smoke patterns in calm interiors.

**Interleaved with atmosphere:** Smoke is updated every atmosphere substep (~50×/tick), not once per tick. This lets smoke ride the shockwave in real time. `dt_scale` amplifies the effect for more dramatic visuals.

**All parameters live on the C++ `SmokeDynamics` class**, set from config at init:

| Parameter | Config key | Value | Notes |
|---|---|---|---|
| `d_smoke` | `physics.d_smoke` | 0.1 | Base smoke diffusion (low = holds shape) |
| `advection_rate` | `physics.advection_rate` | 100.0 | Wind advection strength |
| `dt_scale` | `physics.smoke_dt_scale` | 3.0 | Time multiplier (amplifies smoke response) |
| `wind_diffusion_scale` | `physics.wind_diffusion_scale` | 50.0 | Wind-dependent diffusion: D *= (1 + scale * |wind|²) |

**Sources:** Fire emits smoke into adjacent air tiles (`FIRE_SMOKE_EMISSION * dt * fire_intensity`). Explosions deposit initial smoke cloud with per-tile random variation (0.4–1.0× base). **Known issue:** the noise is too subtle — most tiles near the center saturate at 1.0 regardless of noise, and diffusion smooths the edges within a few substeps. Needs more dramatic variation (e.g. missing patches, larger-scale noise) to create visible texture for the advection to grip.

**Stability note:** No hard CFL limit on the advection term. Practical limit: `advection_rate * actual_dt * |wind| * |grad_smoke| < 1.0` per step to avoid oscillation. The wind-dependent diffusion counteracts this by smoothing high-frequency modes in windy areas.

### 6.5 Fire Simulation

Fire lives on flammable wall tiles. Intensity 0.0 (no fire) to 1.0 (full blaze).

**Current implementation** (C++ `FireSimulation` class, called once per tick):

1. **Wind-biased spreading** — fire computes its own wind from `grad(atmosphere)` and uses it to bias ignition direction (up to 3x boost downwind).
2. **Wind modulates intensity** — `fire += dt * k_wind_net * wind_speed * (fire - k_wind_thresh * wind_speed)`. Strong wind + weak fire = blown out. Strong wind + strong fire = burns hotter.
3. **Intensity growth** — burning tiles grow toward 1.0 at 0.5/s.
4. **Flammable constraint** — fire zeroed on non-flammable tiles.
5. **O2 check** — average atmosphere in 4-connected air neighbors must exceed threshold. Below: fire extinguished.
6. **O2 consumption** — fire reduces atmosphere in adjacent air tiles.
7. **Smoke emission** — fire adds smoke to adjacent air tiles.
8. **Wall damage** — fire reduces `wall_hp`. At 0: `destroy_wall()`.

**Parameters** (set on C++ `FireParams`, hardcoded in Python — **TODO: move to config.toml**):

| Parameter | Value | Description |
|---|---|---|
| `FIRE_D` | 0.3 | Spread rate to neighbors |
| `FIRE_O2_THRESHOLD` | 0.60 | Min atmosphere for fire survival |
| `FIRE_O2_CONSUMPTION` | 0.3 | Atmosphere consumed per step |
| `FIRE_SMOKE_EMISSION` | 0.8 | Smoke produced per step |
| `FIRE_WALL_DAMAGE` | 0.4 | HP damage per step |
| `K_WIND_THRESH` | 0.5 | Fire must exceed this × wind_speed to survive |
| `K_WIND_NET` | 3.0 | Rate of wind effect on intensity |

**Known issues — to be addressed in fire overhaul:**
- Fire computes its own wind internally instead of using precomputed `wind_x`/`wind_y`
- Fire wind uses `grad(atmosphere)` only — missing `wave_p` contribution (shockwave doesn't affect fire)
- Fire wind has no Neumann BC (reads across walls)
- Parameters hardcoded in Python — should move to config.toml
- Fire logic and principles need review and redesign

### 6.6 Temperature & Heat Conduction (NEW — not yet implemented)

*Full design in `docs/implementation_plan_radiation_temperature.md` Section 1.*

**Decision:** Temperature lives on solid tiles only. Air tiles have no meaningful temperature. Heat transfer across air gaps is handled by the raycaster (radiation), not conduction.

**New field:** `gmap.temperature` (float32, initialized to ambient).

**Material conductivity** added to material properties table:
- Hull: 50.0 (metal — heat spreads fast, whole section glows)
- Wood: 0.15 (slow, local heating)
- Glass: 1.0
- Door: 0.3
- Air: 0.0 (no conduction)

**Heterogeneous diffusion:** Standard heat equation with per-tile conductivity. At material interfaces, use harmonic mean:

```
κ_interface = 2 * κ_A * κ_B / (κ_A + κ_B)
flux(A → B) = κ_interface * (T_B - T_A)
```

**Boundary:** Convective cooling at air-adjacent faces (decay toward ambient).

**Heat sources:** Fire, raycaster (heat radiation), explosions, energy weapons.

**Gameplay:** Laser hits hull → heat conducts fast along metal → reaches wood wall → wood crosses ignition_temp → fire starts. Hull glows grey → orange → white (color lerp on render).

### 6.7 Wind/Fire Interaction (implemented in C++)

Both wind-biased spreading and wind-intensity modulation are implemented in `fire_simulation.cpp`. See §6.5 for details.

The effect depends on the ratio of fire intensity to wind strength:
- Weak wind + weak fire → gentle breeze feeds small flame
- Strong wind + weak fire → blown out (fire below threshold)
- Strong wind + strong fire → burns much hotter
- Explosion shockwave → massive transient wind → small fires blown out, big fires flare up

**Note:** Currently fire computes its own wind from `grad(atmosphere)` only. It should use the precomputed wind field (which includes `wave_p`) — to be fixed in the fire overhaul.

### 6.8 Fluid Simulation

**Decision (2026-03-25):** Pipe + damped velocity model. Shallow water equations
diverge and are not worth the complexity for our use case.

**Prototypes:** `prototypes/fluid_test.py`, `prototypes/fluid_scenarios.py`
(aquarium burst, maze flooding), `prototypes/fluid_tilted_ship.py` (Titanic-style
flooding through sloped corridors). All working and visually convincing.

**Concept:** Water (and later oil) flows over the ship's floor, driven by gravity
and ship tilt. Aquariums break, pipes burst, hull takes on water. Fluid pools in
low areas, sloshes when the ship rocks, and interacts with gameplay.

**Resolution: pixel-level, driven by height map.**

The pipe model is so cheap (~10 ops per tile, dt = 10-16 ms, 1-2 steps per frame)
that we can run it at the same resolution as the pixel art textures. This means:

- The **height map** is the single source of truth for floor geometry
- The **normal map** is generated from the height map (used for lighting)
- The **water sim** uses the same height map as terrain (used for fluid flow)
- Water reacts to every detail in the textures — drain grooves, raised thresholds,
  furniture legs, sloped corridors — automatically

No need for a separate coarse physics grid. At pixel-art resolution (e.g. 320x180),
that's ~57,600 tiles. At 10 ops x 2 steps = ~1.1M operations per frame — trivial
on CPU, microseconds on CUDA.

**Terrain + tilt:**
```
effective_height(x, y, t) = height_map[y, x]
    + tilt_x(t) * (x - center_x) * dx
    + tilt_y(t) * (y - center_y) * dx
```
Ship tilt on both axes. Sinusoidal tilt = rocking. Progressive tilt from hull
damage = Titanic effect (water rushes to the low side through corridors and
doorways). Tested in `fluid_tilted_ship.py` — works beautifully.

**The model (pipe + damped velocity):**
```
surface = height_map + tilt_offset + water_depth
flow_velocity += dt * (-g * grad(surface) - damping * flow_velocity)
water_depth -= dt * div(flow_velocity * water_depth)   # upwind flux
```
Unconditionally stable with damping. No CFL constraint. Mass conserved.
dt = 10-16 ms (1-2 steps per 60fps frame). Upwind flux selection prevents
numerical oscillation.

**Why not shallow water equations:** Tested extensively. Lax-Friedrichs flux
splitting with CFL substeps still diverges at wet/dry boundaries and when
water piles against walls. The momentum conservation isn't worth the instability
for our use case. The pipe model produces visually convincing flooding, pooling,
and flow through corridors — which is all we need.

**Fluid sources:**
- Aquariums (break wall -> water floods out + marine creatures)
- Broken pipes (continuous source)
- Hull breaches taking on water (continuous, pressure-dependent)

**Fluid types:** Water first. Oil planned (flammable — oil + fire = spreading
inferno). Types stored as a tag per tile; stacking (oil floats on water) deferred.

**Gameplay effects (planned):**
- Movement penalty in water (deeper = slower)
- Oil ignites from fire -> spreading fire across floor
- Electrical hazard (water + damaged wiring)
- Blocked corridors if water is deep enough

**Surface rendering (planned):** Augment the water surface with slow sine waves
or a low-speed wave equation to give an ocean/ripple look. Wave speed for surface
waves is sqrt(g * depth) ~ 1-3 m/s (100x cheaper than atmosphere's sound waves).

**Sparse simulation:** Only tiles with fluid need updating. Dry areas skipped.

**Next steps:** Implement in C++ with single-step API matching AtmosphereSolver
/ SmokeDynamics pattern. Test at pixel resolution with actual game textures.

#### Water–Air Coupling: Volume Displacement

_Design idea (2026-03-26)_

The pipe model already simulates water flow. The atmosphere sim already simulates
pressure and gas advection. **Coupling them requires almost no new machinery** —
the water surface defines how much air volume remains in each cell, and pressure
does the rest.

**Why it works — one-directional coupling:**
Water is ~1000× denser than air. Water doesn't care about air (gravity and the
pipe model drive it). Air cares about water (water defines available volume).
Water sim runs first, air sim reacts. No two-phase solver needed.

**The mechanism:**

Each cell has a total capacity (floor-to-ceiling volume). Water occupies some of it.
Air gets the rest:

```
air_volume[i] = cell_capacity[i] - water_depth[i] * cell_area
P[i] = air_mass[i] * R * T / air_volume[i]    # ideal gas law
```

Compress air into smaller volume → pressure rises → pressure difference drives
airflow through openings (doors, holes, ducts). Smoke and gas piggyback on this
flow as passive scalars — exactly as they already do with the wind field.

**Update schedule integration:**

| Step | What happens |
|------|-------------|
| 1 | **Water pipe model** runs (unchanged) — updates `water_depth` per cell |
| 2 | Compute `air_volume = capacity - water_depth × area` per cell |
| 3 | Compute air pressure from ideal gas: `P = nRT/V` |
| 4 | **Atmosphere wave equation** runs, with per-cell pressure influenced by reduced volume |
| 5 | Gas advection runs as normal, riding the wind field |

Steps 2–3 are the only new work. Everything else already exists.

**Motivating scenario:** Ship tilts → water sloshes to starboard (pipe model) →
air volume on starboard shrinks → pressure spikes → smoke and gas get shoved to
port through doors and corridors → crew on port side suddenly gets hit by smoke
they thought was contained. Emergent gameplay from two systems that already exist.

**Implementation notes:**
- **No boundary tracking.** The water surface height IS the boundary — the pipe
  model already computes it per cell per tick.
- **Total pressure stays consistent.** Air mass is conserved (just redistributes).
  Water is incompressible. No mass appears or disappears.
- **Cost is trivial.** Two extra scalar operations per wet cell per tick (volume
  subtraction + pressure recomputation).
- **Fully flooded cell:** `water_depth ≥ cell_capacity` → air volume = 0, cell
  becomes a wall for the atmosphere sim. Physically correct — a flooded corridor
  blocks airflow.
- **Interaction with breaches.** Hull breach lets water IN and air OUT. Vacuum
  relaxation at the breach handles the air side. Water entering via the pipe model
  raises water depth, further compressing air — creating a brief pressure spike
  before air escapes. Dramatic and correct.

**Open questions:**
- Should volume-displacement pressure feed into the wave equation as a source term,
  or run as a separate pressure-equalization pass? (Probably source term — cleaner.)
- At what water depth does a cell become impassable to air? Full ceiling, or a
  threshold (e.g. 90%) for air pockets?
- Oil floats on water — if implemented, does it seal the surface and trap air below?

### 6.9 Physics Step Orchestration

All physics advance in `Physics.step(gmap, sim_time)`:

**Current implementation (IMEX, Python-orchestrated):** Single substep loop calls C++ solvers. Atmosphere and smoke are interleaved so smoke rides the shockwave.

```
sim_time = 1 / ticks_per_second = 83.3 ms per game tick
dt = AtmosphereSolver.max_dt() ≈ 1.67 ms (wave CFL only)
n = ceil(sim_time / dt) ≈ 50 substeps

for each substep:
   1. AtmosphereSolver.step(dt)     — wave kick + transfer + implicit diffusion + BCs + wind
   2. SmokeDynamics.step(dt * scale) — diffusion + advection by precomputed wind

after all substeps:
   3. FireSimulation.step(sim_time)  — spreading, O2, smoke emission, wall damage
   4. Raycaster.update_from_fire()   — light map from fire sources

4. Fire step: single step at full sim_time
   - Spread, wind bias, O2 check, consumption, smoke emission, wall damage

5. Temperature step (NEW): substeps based on max conductivity CFL
   - Heterogeneous diffusion on solid tiles

6. Raycaster (NEW): per light source
   - Update light_map and heat deposits
```

**C++ target (interleaved time advancement):** Wave and diffusion operate on the same atmosphere field but at different substep sizes (dt_wave=2.17ms, dt_diff=1.2ms). Running all wave substeps first means diffusion sees the final atmosphere from the entire wave pass, rather than the gradual energy deposition. The physically correct approach is to interleave them:

```
t_wave = 0, t_diff = 0
while t_wave < sim_time or t_diff < sim_time:
    if t_wave <= t_diff:
        wave_substep(dt_wave)       // deposits into atmosphere
        t_wave += dt_wave
    else:
        diffusion_substep(dt_diff)  // spreads atmosphere
        t_diff += dt_diff

// Then single-step systems that read the final state:
smoke_step(sim_time)
fire_step(sim_time)
temperature_step(sim_time)    // substeps internally
raycaster_update()
```

Both systems advance through the same 83.3ms but stay within one substep of each other in simulated time. Atmosphere energy from the wave starts diffusing immediately rather than waiting for all wave substeps to complete. Same total substep count — no extra compute cost. Smoke, fire, and temperature still run after the coupled wave+diffusion pass since they only need the final atmospheric state for the tick.

### 6.10 Double-Buffered Propagation (TODO — needs design before implementation)

Within each system's substeps, the Laplacian reads neighbor values and writes updated values. In C++ with raw tile-by-tile loops, tiles updated early in the loop produce different neighbor values for tiles updated later — an order-of-operations bug. Double buffering fixes this:

1. Each system's primary array has a *current* buffer (read-only during the substep) and a *next* buffer (write target).
2. The substep reads from *current* and writes to *next*.
3. After each substep, *next* becomes *current* (pointer swap).

**Important nuance:** Double buffering applies **within each system's substeps**, not across the whole tick. The cross-system pipeline (wave → atmosphere → smoke → fire) is intentionally sequential — wave deposits energy into atmosphere, then diffusion spreads it. Double-buffering at the tick level would break this coupling.

**Python prototype:** Not an issue today. NumPy bulk array operations (`wave_p += c2 * laplacian * dt`) read the entire array before writing — effectively atomic. No per-tile ordering exists.

**C++ port:** Needs double buffering within each system's substep loop. The exact design (which arrays need current/next pairs, how cross-system writes like wave→atmosphere interact with the buffers) needs to be worked out before implementation.

---

## 7. 2D Raycasting System (NEW)

*Full design in `docs/implementation_plan_radiation_temperature.md` Section 3.*

### Architecture

One generic raycaster function. Sources define what they emit (light, heat, damage). The raycaster marches rays tile-by-tile using DDA, depositing at each tile and accumulating absorption.

### Source Profiles

Each light source has configurable properties:

```
position, max_range, ray_count, angle_center, angle_spread,
intensity, color, heat, jitter, falloff_fn
```

**Prebuilt profiles:** point_light, fire, flashlight, energy_weapon, muzzle_flash, emergency_light. All values overridable per instance.

**Ray count:** Default `ceil(2 * pi * max_range)`, scaled for cones, overridable. A 40-degree flashlight at range 30 needs only ~22 rays.

**Jitter:** Random angular offset per ray per cast. Fire sources use ~3 degrees of jitter for flickering shadow edges. Regenerated each cast.

### Ray March (DDA)

Each ray walks tile-by-tile:
1. Deposit light/heat at current tile (with distance falloff)
2. If wall: stop (or reflect if bounces remain)
3. If smoke: attenuate by `1 - smoke_density * absorption_rate`
4. If unit: absorb most radiation (important for energy weapons, heat damage)
5. Step to next tile (DDA: exact, never skips a tile)

### Integration

- Fire tiles are light sources (fire profile, emits heat)
- `light_map[y, x]` drives stealth mechanic (below threshold = in shadow)
- Smoke + light along ray path = volumetric light shafts (god ray approximation)
- Energy weapons use raycaster with narrow cone, high heat

### Reflection (secondary priority)

Material reflection coefficient (hull: 0.9, wood: 0.3). Ray bounces off wall with reduced intensity. Max 1 bounce for prototype. Metal corridors bounce flashlight beams.

### Cost

Worst case (20 sources): ~40,000 tile operations. Comparable to existing physics substeps. In C++: sub-millisecond.

---

## 8. Lightning Bolt Effect (NEW)

*Full design in `docs/implementation_plan_radiation_temperature.md` Section 4.*

**Trigger:** Event-driven (damaged electronics, weapon impact, exposed wiring). Not continuous.

**Target selection:** Search within radius, prefer by conductivity (metal > water > unit > random air fallback).

**Visual path:** Recursive midpoint displacement (depth 5-6, ~33-65 points). Regenerated each frame for 2-3 frames — bolt appears to crackle and dance.

**Damage:** Units on or adjacent to bolt path take electrical damage. Future: bolt hits water tile → flood-fill connected water → all units on wet tiles damaged.

**Cost:** Negligible (~60 point calculations + 60 line segments per bolt).

---

## 9. Units & AI

### Entity List (architectural principle)

Mobile entities (squad members, creatures, items) are stored in a **separate entity list**, not on the spatial grid. Entities carry properties that don't map well to spatial grids: inventory, action points, allegiance, AI state, status effects. Their position is an `(fx, fy)` index into the tile grid.

This separation is already how the Python prototype works (units are objects in a list with grid coordinates), but naming it as a principle ensures the C++ port maintains it: `std::vector<Unit>` alongside the `GameMap`, not embedded in tiles.

**Bridge to spatial grid:** Each tick, unit positions are stamped into the `obstacles` cached array via `stamp_units()`. This makes units act as walls for all physics systems — waves reflect off them, smoke and gas flow around them, light is blocked. Units live in the list, but their physical presence is projected onto the grid.

### Current: Single Unit Class

Currently one `Unit` class serves both marines and zombies. This causes field bloat:

- Marines don't use: `zombie_activated`, `zombie_path`, `zombie_path_idx`, `zombie_move_accumulator`, `last_melee_tick`, `killed_by_zombie`
- Zombies don't use: `orders`, `ap` (but see inventory note below)

### Target: Separated Unit Types

**Base fields (all units):**
```
name, team, fx, fy, alive, hp, max_hp, facing, inventory[]
```

Note: the current code also has `fxf, fyf` (float position for interpolation). Per the v2 design, these should be removed from game state — units are always on integer tile positions. The renderer handles interpolation between ticks.

**Inventory is a base field, not marine-specific.** When a marine is converted to a zombie, it retains its inventory (grenades, explosives). Zombies can't *use* items, but they still *carry* them. This creates emergent interactions: converted zombie carries a grenade → walks through fire → grenade overheats past ignition_temp → detonation. No special-case code — the temperature system and explosion system handle it naturally.

**Marine-specific:**
```
orders[], ap[2], move_path[], last_fire_tick, fire_target
```

**Zombie-specific:**
```
zombie_activated, zombie_path[], zombie_path_idx, zombie_move_accumulator,
last_melee_tick, speed (replaces zombie_speed_override hack)
```

**Zombie as state, not type.** A zombie is any unit that has been converted — it switches to zombie AI but retains its base fields and inventory. When zombified: unit uses zombie AI (activation, pathfinding toward nearest living player, melee), ignores orders/AP, and cannot use items.

**Unit type architecture: TBD.** The game will have many entity types beyond marines and zombies — robots, animals, worms, etc. Some may need specialized representations (e.g. a worm could store body segment positions in a ring buffer — compute new head position, recycle tail slot, the rest shift automatically). Whether this means one flexible Unit class, a type hierarchy, or composition is an open design question. Decide per-entity as they're implemented, not upfront.

### Zombie Variants (current)

| Variant | HP | Speed (ticks/tile) | Notes |
|---|---|---|---|
| Regular | 400 | 7 | From config |
| Runner | 100 | 4 | `hp // 4`, `speed - 3` |
| Brute | 1200 | 10 | `hp * 3`, `speed + 3` |

Speed is currently set via `zombie_speed_override` attribute (fragile). Should be a proper field or config table.

### Zombie AI

1. **Trigger detection:** Each inactive zombie checks LOS to players within `trigger_radius` (24 tiles).
2. **Chain activation:** BFS propagation — activated zombies activate nearby inactive zombies within `propagation_radius` (15 tiles).
3. **Target selection:** Nearest living player (by Euclidean distance).
4. **Movement:** A* pathfinding toward target. Repath every 5 steps to track moving players. Speed: one tile move per `ticks_per_tile` ticks.
5. **Melee attack:** When adjacent (distance <= tile_size + 1). Damage: 60 HP per hit. Cooldown: 12 ticks.
6. **Zombie conversion:** Marines killed by zombies become zombies at end of round.

---

## 10. Turn System & Execution

### Game States

```
STATE_PLANNING  →  (Space/Enter)  →  STATE_EXECUTING  →  (all ticks done)  →  STATE_PLANNING
```

### Turn Structure

| Property | Config key | Value |
|---|---|---|
| Ticks per second | `clock.ticks_per_second` | 12 |
| Phases per round | `clock.phases_per_round` | 2 |
| Phase duration | `clock.phase_duration_seconds` | 5.0 s |
| Ticks per phase | derived | 60 |
| Ticks per round | derived | 120 |
| AP per phase | `clock.ap_per_phase` | 2 |

One round = 2 phases = 10 seconds = 120 ticks of game time.

### Planning Phase

Player assigns orders to marines for each phase (Phase 1 and Phase 2, toggled with Tab):

- **Movement orders** (Move & Attack / Move w/ Cover / Sprint): click to place waypoints. No AP cost. Multiple waypoints per phase.
- **Grenade order** (G): click target, scroll to set fuse timer. Costs 1 AP.
- **Door Explosive order** (B): click adjacent tile, scroll to set detonation slot (Start P1 / Between P1-P2 / End P2). Costs 1 AP.
- **Fire order** (F): click target tile. Costs 1 AP. Unit fires burst at target during execution.
- **Backspace**: undo last order (refunds AP and inventory).

### Execution Phase

1. **Setup:** Compute A* paths for all player movement. Spawn grenade projectiles. Process DET_START_PHASE1 door explosives.
2. **Tick loop:** Accumulate real time → convert to game ticks → process each tick:
   - Update projectile positions, check detonations
   - Update player unit positions from precomputed paths
   - Process shooting (fire orders + move & attack auto-fire)
   - Update zombie AI (activation, movement, melee)
   - Stamp unit positions into obstacles
   - Run physics step
3. **Phase transition:** At tick 60, process DET_BETWEEN_PHASES explosives.
4. **End:** At tick 120, process DET_END_PHASE2 explosives. Zombie conversion. Snap positions. Clear orders. Return to planning.

### Playback Speed

+/- during execution adjusts `exec_speed` (0.25x to 10x). This scales the rate at which game ticks are consumed from real time. The simulation itself is unchanged — same number of ticks, same physics.

---

## 11. Combat

### Rifle

**Burst fire:** 5 bullets per burst, 2 ticks between bursts. Each bullet is a ray march at `base_angle + random_offset` (cone half-angle: 3 degrees). Ray stops at wall hit or unit hit.

**Damage:** 10 per bullet to marines. Zombies take `damage * bullet_damage_multiplier` (0.25 = 2.5 per bullet). Fire damage to zombies uses `fire_damage_multiplier` (4.0).

**LOS check:** Bresenham ray from shooter center to target. Blocked by `is_wall`.

**Auto-fire:** In Move & Attack mode, marines automatically fire at nearest visible enemy each burst interval.

**Range:** 90 tiles (30 meters).

### Explosions

**Grenade:** radius 6, pressure 10.0, wall_damage 200, unit_damage 60. Thrown as projectile with configurable fuse (0-10s, default 0.5s).

**Door explosive:** radius 3, pressure 5.0, wall_damage 500, unit_damage 60. Detonates at a scheduled slot (start P1 / between phases / end P2).

**Explosion effects:**
- Wall damage with distance falloff. Walls destroyed when HP reaches 0.
- Wave source deposit (propagated by wave equation over substeps).
- Atmosphere boost (0.3 * pressure * falloff — creates sustained wind).
- Smoke clearing (inner 40% of radius).
- Fire ignition (flammable tiles within 70% of radius, intensity = 0.5 * falloff).
- Unit blast damage (falloff from center, threshold: 5 damage minimum).

### Blast Damage Threshold

Units take blast damage only if `damage >= blast_damage_threshold` (5). Prevents chip damage from distant explosions.

---

## 12. Pathfinding

### Standard A* (used by zombies and player unit path computation)

8-directional movement on the tile grid. Alternating diagonal cost (D&D 3.5 style: first diagonal costs 1, second costs 2, alternating). Node expansion capped at 50,000.

**Blocking check:** `is_passable_block(y, x)` — tests if the entire 3x3 unit block at that position is passable.

### Temporal A* (implemented but NOT yet used)

Adds time dimension: state is `(x, y, tick)`. Uses `ReservationTable` to avoid collisions between player units. Supports wait actions. **Currently unused** — player units can overlap during execution. Should be enabled when multi-unit coordination is needed.

### ReservationTable

Sparse `{(x, y, tick): unit_id}` map. Reserves all tiles in a unit's 3x3 block for the duration of each path segment. Supports per-unit clearing and exclusion checks.

---

## 13. Rendering

### Current: Pygame

Rendering is cleanly separated from game logic. Each subsystem has its own draw method. Draw order:

1. `_draw_map()` — material grid as colored rectangles + grid lines
2. `_draw_atmosphere()` — pressure overlay (fire color ramp for overpressure, blue-purple for underpressure)
3. `_draw_smoke()` — gray semitransparent overlay
4. `_draw_fire()` — orange/yellow glow overlay
5. `_draw_orders()` — waypoint lines, targeting rings, grenade/explosive markers (planning only)
6. `_draw_projectiles()` — grenade circles in flight
7. `_draw_units()` — marine sprites (8-directional) + zombie rectangles + HP bars + name labels
8. `_draw_shots()` — bullet tracers (fade over duration)
9. `_draw_ui_panel()` — side panel: turn info, mode selector, unit details, order timeline, performance stats, controls help
10. `_draw_cursor_info()` — ghost placement indicator, crosshairs (planning only)

### Color Schemes

**Pressure overlay (overpressure — fire ramp):**
- t=0.00-0.25: black → red
- t=0.25-0.50: red → orange (R=255, G: 0→140)
- t=0.50-0.75: orange → yellow (G: 140→255, B: 0→80)
- t=0.75-1.00: yellow → white (B: 80→255)
- Alpha: `excess * 80`, capped at 255

**Pressure overlay (underpressure — vacuum):**
- Blue-purple: R=60+40*deficit, G=40, B=255
- Alpha: `deficit * 400`, capped at 220

**Smoke:** R=180, G=160, B=140, A=`density * 220` (capped at 200)

**Fire glow:** R=150+105t, G=80t, B=20t, A=180+75t (dark red → orange → bright yellow)

**Shot tracers:** White-yellow, fading over `shot_tracer_duration` (0.4s). Muzzle flash circle for first 50ms.

### Target: Raylib (C++)

Raylib is the intended rendering backend for the C++ version. The clean render separation in the current code maps directly — each draw method becomes a Raylib draw call sequence.

---

## 14. Configuration & Hot-Reload

### Current: config.toml + GameConfig class

`config.toml` is loaded at startup into a `GameConfig` object with nested `Namespace` attribute access (`CFG.clock.ticks_per_second`). `CFG.reload()` re-reads the file (bound to F5).

**Derived values** computed on load:
- `fine_w = map_w * coarse` (tiles_per_unit, historically called "coarse")
- `fine_h = map_h * coarse`
- `coarse_px = fine_tile_px * coarse`
- `ticks_per_phase = ticks_per_second * phase_duration_seconds`
- `ticks_per_round = ticks_per_phase * phases_per_round`

### Known Issue: Inconsistent Parameter Locations

Many tunable parameters are **not** in config.toml:

| Parameter | Location | Should be |
|---|---|---|
| `WAVE_C`, `WAVE_DAMPING`, etc. | Physics class constants | config.toml `[physics.wave]` |
| `FIRE_D`, `FIRE_O2_THRESHOLD`, etc. | Physics class constants | config.toml `[physics.fire]` |
| Smoke wave advection (80.0) | Inline magic number | config.toml `[physics.smoke]` |
| Explosion atmosphere boost (0.3) | Inline magic number | config.toml `[weapons.grenade]` |
| Smoke clearing radius (0.4) | Inline magic number | config.toml `[weapons.grenade]` |
| Fire ignition radius (0.7) | Inline magic number | config.toml `[weapons.grenade]` |

**Target:** All tunable parameters in config.toml. No hardcoded constants. No magic numbers.

### C++ Config Strategy

Option A: Keep TOML in Python, pass values to C++ on reload via pybind11.
Option B: Use a C++ TOML parser (e.g., toml11) and load directly.

Either way, hot-reload must work — it's essential for iteration.

---

## 15. C++ Port Strategy

### Transition Architecture

```
Python (pygame)          C++ (pybind11 module)
┌──────────────┐        ┌──────────────────┐
│  Renderer    │◄───────│  GameMap arrays   │  (shared memory, zero-copy)
│  Input       │        │  Physics.step()   │
│  Config      │───────►│  Raycaster        │
│  Game loop   │        │  Pathfinding      │
└──────────────┘        └──────────────────┘
```

numpy arrays and C++ arrays share the **exact same memory** via pybind11's buffer protocol. Pygame reads the arrays it already uses. The Python game loop calls C++ `step()` instead of Python `Physics.step()`.

### Port Order (by performance impact)

1. **Wave equation** — biggest bottleneck (~39 substeps of full-grid Laplacian per tick)
2. **Atmosphere diffusion** — ~70 substeps per tick
3. **Fire simulation** — O(n) neighbor loops, wall destruction
4. **Smoke dynamics** — one step but includes gradient computation
5. **Raycaster** — new system, implement directly in C++
6. **Temperature diffusion** — new system, implement directly in C++
7. **Pathfinding** — A* with 50k node limit, heap operations
8. **Rendering** (final step) — replace pygame with Raylib

### Build Targets

The C++ simulation compiles into two targets from the **same source code**:

**Engine Plugin** — Compiled as a module within the game engine project. The presentation layer calls directly into the simulation via C++ APIs.

**Python Module (.so / .pyd)** — Built with CMake + pybind11. Produces an importable Python package for headless training, evaluation, and analysis. `env.state.pressure` returns a standard NumPy array that can be fed directly into a neural network.

This ensures that the game and the training environment run *identical* simulation logic — no translation bugs, no drift between the two.

### C++ Class Structure (target)

```cpp
// Grid2D<T> — templated 2D array, contiguous memory, pybind11-compatible
template<typename T>
class Grid2D {
    int width, height;
    std::vector<T> data;  // row-major
    T& operator()(int y, int x);
    T* raw();  // for pybind11 buffer
};

// GameMap — world state (all fields)
class GameMap {
    Grid2D<int8_t> material;
    Grid2D<float> wall_hp, atmosphere, wave_p, wave_v, wave_source;
    Grid2D<float> smoke, fire, temperature, light_map;
    Grid2D<bool> obstacles, is_wall, is_vacuum, flammable;
    void stamp_units(const std::vector<Unit>& units);
    void destroy_wall(int y, int x);
    void update_caches();
};

// Physics subsystems — each owns its parameters, operates on GameMap
class WaveEquationSolver {
    float c, damping, transfer_rate, source_feed_rate;
    void step(GameMap& map, float sim_time);
};

class AtmosphereDiffusion {
    float d_atm;
    void step(GameMap& map, float sim_time);
};

class SmokeDynamics {
    float d_smoke, advection_rate, wave_advection_rate;
    void step(GameMap& map, float sim_time);
};

class FireSimulation {
    float spread_rate, o2_threshold, o2_consumption;
    float smoke_emission, wall_damage;
    float k_cool, k_o2;  // wind interaction
    void step(GameMap& map, float sim_time);
};

class TemperatureSolver {
    float h_conv, t_ambient;
    void step(GameMap& map, float sim_time);
};

class Raycaster {
    void cast_all(GameMap& map, const std::vector<LightSource>& sources);
};

// PhysicsEngine — orchestrates all subsystems
class PhysicsEngine {
    WaveEquationSolver wave;
    AtmosphereDiffusion atmosphere;
    SmokeDynamics smoke;
    FireSimulation fire;
    TemperatureSolver temperature;
    Raycaster raycaster;
    void step(GameMap& map, float sim_time);
};
```

---

## 16. Known Issues & Refactoring Targets

### High Priority (fix before or during port)

1. **Config inconsistency** — wave, fire, and explosion parameters hardcoded instead of in config.toml. Unify all tunable parameters.

2. **Game class is a God Object** (~1300 lines) — input handling, execution loop, combat, zombie AI, rendering all in one class. Target split: GameState, InputHandler, SimulationDriver, Renderer, ZombieAI.

3. **Unit class mixes Marine and Zombie** — field bloat, zombie_speed_override hack. Separate into distinct types.

### Medium Priority

4. **Temporal A* unused** — `ReservationTable` and `temporal_astar` exist but are never called. Player units can overlap during execution. Enable or remove.

5. **Fire wall destruction double-loop** — `for fy ... for fx ... if burned_out[fy, fx]` should use `np.argwhere(burned_out)`.

6. **Smoke/fire have no substeps** — they run once per tick at 83ms dt. May cause instability with large parameter values. Evaluate whether substeps are needed.

7. **Phase transition detection** — `new_phase = exec_tick // tpp` is correct but fragile. Consider explicit phase boundary tracking.

### Low Priority (nice to have)

8. **Pathfinding constants hardcoded** — `FINE_W = 120`, `FINE_H = 75`, `UNIT_SIZE = 3` in pathfinding.py don't reference config.

9. **UI panel** — `_draw_ui_panel()` is 207 lines. Could use a lightweight UI layout system.

10. **Float positions should be removed** — `fxf/fyf` exist in Unit but per `design_v2_turn_and_combat_overhaul.md`, units should always be on integer tile positions. Movement is tile-to-tile hops. Rendering interpolates between ticks for smooth animation — this is a renderer concern, not game state.

11. **Liquids system not yet implemented** — Liquid type (none, blood, water, fuel) and liquid depth are planned per-tile state fields. Liquids interact with fire (water extinguishes, fuel accelerates), creatures (blood attracts), and electricity (lightning conducts through water). Needs its own design pass before implementation.

12. **Double-buffered propagation not yet implemented** — See §6.9. Should be introduced during C++ port.

---

## 17. ML & Neural Network Considerations

The array-based state representation is designed to be directly consumable by convolutional neural networks, following the stacked feature-plane approach used by AlphaStar (StarCraft II).

- Each cached array becomes one (or more) input channels — material, atmosphere, smoke, fire, temperature, light, etc. stacked into a 3D tensor `[C, H, W]`.
- The headless simulation (§2 Two-Layer Architecture) enables massively parallel self-play without rendering overhead.
- Action space definition and reward shaping are deferred to a future design document.
- The same C++ simulation compiles to both a game engine plugin and a Python module (§15 Build Targets), ensuring training and gameplay run identical logic.

A typical training loop using the Gymnasium interface:

```python
import sim                          # compiled C++ module

env = sim.BoardingEnv(map="frigate_01")
obs = env.reset()                   # returns dict of NumPy arrays (zero-copy)

for step in range(max_steps):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    agent.learn(obs, reward, done)
```

This is compatible with standard RL libraries (Stable Baselines3, RLlib, CleanRL, etc.).

---

## 18. Design Document Roadmap

Each system is specified in its own design document. Documents progress through: **brainstorm** (markdown notes) → **design document** → **implementation**.

### Environmental Simulation

| Doc | System | Status |
|-----|--------|--------|
| 01 | Structure, Materials & Destruction | **Implemented** (game.py) |
| 02 | Pressure & Decompression | **Implemented** (atmosphere diffusion) |
| 03 | Fire Propagation | **Implemented** (fire system) |
| 04 | Liquids (blood, water, fuel) | Not started |
| 05 | Smoke Propagation | **Implemented** (diffusion + advection) |
| 06 | Explosions | **Implemented** (grenades, door explosives, wave eq) |
| 07 | Line of Sight & Cover | Partially implemented (Bresenham LOS; raycaster designed) |

### New Systems (from implementation plan)

| System | Status |
|--------|--------|
| Temperature & Heat Conduction | **Designed** (implementation_plan_radiation_temperature.md) |
| Wind/Fire Interaction | **Designed** (implementation_plan_radiation_temperature.md) |
| 2D Raycasting (light, heat, weapons) | **Designed** (implementation_plan_radiation_temperature.md) |
| Lightning Bolts | **Designed** (implementation_plan_radiation_temperature.md) |

### Game Mechanics

| Doc | System | Status |
|-----|--------|--------|
| 08 | Turn System & Simultaneous Turns | **Implemented** (design_v2 + game.py) |
| 09 | Combat Mechanics | **Implemented** (rifle, explosions, melee) |
| 10 | Economy & Currency | Not started |
| 11 | Inventory & Equipment | Partial (grenades, explosives as booleans) |
| 12 | Player Interface & Interaction | **Implemented** (pygame UI) |

### Presentation

| Doc | System | Status |
|-----|--------|--------|
| 13 | Graphics, Lighting & Art Pipeline | **Designed** (raycasting in implementation plan) |

### Content Design

| Doc | System | Status |
|-----|--------|--------|
| 14 | Creatures & AI Behaviors | Partial (zombie AI, 3 variants) |
| 15 | Missions & Objectives | Brainstorm (missions.md) |
| 16 | Ship Layouts & Level Design | Not started (test map only) |
| 17 | Narrative & Lore | In progress (missions.md, lore files) |

### Infrastructure

| System | Status |
|--------|--------|
| C++ Port (pybind11 + physics) | Not started (architecture designed) |
| Neural Network Training Pipeline | Not started (architecture designed) |
| Double-Buffered Propagation | Not started (introduce during C++ port) |
| Tile Objects (authoritative state) | Not started (introduce during C++ port) |
