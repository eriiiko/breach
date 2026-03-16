# Implementation Plan: Radiation, Temperature & Lighting Systems

*Design decisions from discussion 2026-03-16. Intended to guide both Python prototype and C++ implementation.*

---

## 1. Temperature Field & Heat Conduction (Solids Only)

### Decision

Temperature lives on **solid tiles only** (hull, wood, door, glass, etc.). Air tiles do not have a meaningful temperature — heat transfer across air gaps is handled by the raycaster (Section 2), which is the physically correct mechanism (radiation, not conduction).

### New field

```python
gmap.temperature = np.zeros((fh, fw), dtype=np.float32)  # Kelvin (or arbitrary units)
```

Initialized to ambient (e.g. 293 K). Only meaningful where `gmap.is_wall == True`.

### Material properties — add conductivity

```python
# Add to material properties (config.toml or MATERIAL_PROPERTIES dict):
#   conductivity: thermal diffusivity (how fast heat spreads through this material)
#   ignition_temp: temperature at which this material catches fire (if flammable)
#
# Approximate relative values (not real-world units, tuned for gameplay):
#   MAT_HULL:  conductivity = 50.0   (metal — heat spreads fast, entire hull section glows)
#   MAT_WOOD:  conductivity = 0.15   (wood — slow, stays local, eventually ignites)
#   MAT_GLASS: conductivity = 1.0    (between metal and wood)
#   MAT_DOOR:  conductivity = 0.3    (wood-like)
#   MAT_AIR:   conductivity = 0.0    (not used — air tiles skipped)
```

### Heterogeneous diffusion equation

Standard heat equation, but conductivity varies per tile. At the interface between two materials, use the **harmonic mean** of their conductivities — this ensures physically correct flux continuity:

```
flux(A → B) = κ_interface * (T_B - T_A)

κ_interface = 2 * κ_A * κ_B / (κ_A + κ_B)    # harmonic mean
```

If either κ is 0 (air boundary), flux is 0 through conduction. Heat only crosses air via radiation (raycaster).

#### Discrete update (per substep)

```python
def step_temperature(gmap, dt):
    T = gmap.temperature
    κ = gmap.conductivity          # per-tile, derived from material

    for direction in [up, down, left, right]:
        T_neighbor = roll(T, direction)
        κ_neighbor = roll(κ, direction)
        κ_interface = 2.0 * κ * κ_neighbor / (κ + κ_neighbor + epsilon)
        flux += κ_interface * (T_neighbor - T)

    T += dt * flux

    # Boundary: convective cooling at air-adjacent faces
    # (decay toward ambient where neighbor is air)
    air_neighbor_count = count of non-wall neighbors
    T -= dt * h_conv * air_neighbor_count * (T - T_ambient)

    # Only solid tiles hold temperature
    T[~gmap.is_wall] = T_ambient

    # Visual: color lerp based on temperature
    # grey → orange → white as T increases (for hull tiles)
```

#### Stability

Same CFL condition as other diffusion systems: `dt < dx² / (4 * κ_max)`. With κ_max = 50 for metal, this may need a few substeps. In C++ this is negligible cost.

### Heat sources

- **Fire**: burning tiles deposit heat into adjacent solid tiles (already touching via wall geometry)
- **Raycaster**: heat radiation deposits at target tiles (Section 2)
- **Explosions**: brief intense heat at blast center (already ignites flammable tiles)
- **External attacks**: laser/energy weapons deposit heat at impact tile

### Gameplay payoffs

- Laser hits hull tile → heat conducts fast along connected metal → reaches wood interior wall → wood crosses ignition_temp → fire starts (emergent chain, no scripting)
- Hull section glows grey → orange → white based on temperature (color lerp on render, nearly free)
- Metal door heated from one side eventually conducts through → heats other side
- Can skip conduction for some materials: set `conductivity = 0` and they act as insulators

---

## 2. Wind/Fire Interaction

### Decision

Modeled directly on fire tiles using existing wind (atmosphere gradient). No air temperature field needed.

### Current state (game.py)

Fire already has wind-biased *spreading* (lines 451-461) — the atmosphere gradient steers which direction fire ignites neighbors. But there's no mechanism for wind to **extinguish** or **intensify** existing fires.

### New mechanic: wind modulates fire intensity

Add to `step_fire()`, after the spreading logic:

```python
# Wind speed at each tile (magnitude of atmosphere gradient)
wind_x = -(np.roll(gmap.atmosphere, -1, axis=1) -
            np.roll(gmap.atmosphere, 1, axis=1)) / 2.0
wind_y = -(np.roll(gmap.atmosphere, -1, axis=0) -
            np.roll(gmap.atmosphere, 1, axis=0)) / 2.0
wind_speed = np.sqrt(wind_x**2 + wind_y**2)

# Two competing effects on burning tiles:
burning = gmap.fire > 0.01

# Cooling: weak fire loses heat easily (no thermal mass to resist wind)
cooling = K_COOL * wind_speed * (1.0 - gmap.fire)

# O2 boost: strong fire has more fuel, wind feeds oxygen to flames
o2_boost = K_O2 * wind_speed * gmap.fire

# Net effect
gmap.fire[burning] += dt * (o2_boost[burning] - cooling[burning])

# If fire drops below threshold, extinguish
gmap.fire[gmap.fire < 0.01] = 0.0
```

### Behavior

- **Weak fire + strong wind**: cooling >> O2 boost → fire extinguished
- **Strong fire + strong wind**: O2 boost >> cooling → fire grows hotter
- **No wind**: neither term contributes → fire behaves as current (grows toward 1.0)
- **Explosion shockwave**: creates massive transient wind → small fires blown out, big fires flare up

### Parameters to tune

- `K_COOL`: convective cooling coefficient
- `K_O2`: oxygen feeding coefficient
- The crossover point (where wind helps vs. hurts) is determined by fire intensity alone

### Connection to explosions

Explosions already do two things:
1. `wave_p` — fast shockwave (wave equation, milliseconds)
2. `atmosphere` perturbation — sustained wind (diffusion, seconds)

Both create atmosphere gradients. The shockwave creates a brief intense wind spike (blows out candles). The atmosphere perturbation creates sustained airflow (fans large fires). This is already in the code — the wind/fire interaction just reads the gradient that's already there.

---

## 3. 2D Raycasting System

### Decision

Fixed-angle ray marching with per-source emission profiles. One generic raycaster function, multiple source types defined by configuration. Chosen over edge-casting because:

- Rays naturally carry directional intensity (flashlight brighter in center, dimmer at edges)
- Smoke absorption requires per-tile traversal along the ray path — edge-casting gives polygons, not paths
- Heat/damage deposition along the ray path requires visiting each tile
- Reflection is a natural extension (bounce and keep marching)
- Simple to implement, predictable cost

### Terminology note

All references to "tiles" mean the base spatial grid (1/3 m side). There is no separate coarse grid — marines occupy 3x3 tiles, but the raycaster operates at tile resolution like all other systems.

### Architecture overview

```
LightSource  →  Raycaster  →  LightMap (per-tile intensity + color)
                           →  HeatMap deposits (into gmap.temperature)
                           →  DamageMap deposits (for energy weapons)
```

The raycaster is a single generic function. The source defines what it emits. The target tile receives whatever the source is emitting, attenuated by distance and obstacles.

### LightSource definition

```python
class LightSource:
    x: int                    # tile position
    y: int
    max_range: int            # max distance in tiles (determines default ray_count)
    ray_count: int            # configurable, default = ceil(2π * max_range)
    angle_center: float       # facing direction (radians, 0 = right)
    angle_spread: float       # cone width (radians, 2π = omnidirectional)
    intensity: float          # base brightness (arbitrary units)
    color: (float, float, float)  # RGB, normalized
    heat: float               # heat emission multiplier (0 = no heat, 1.0 = full)
    jitter: float             # angular jitter per ray per cast (radians, 0 = none)
    falloff_fn: str           # "uniform", "cosine", "sharp" — angular intensity profile
```

### Prebuilt source profiles

```python
LIGHT_PROFILES = {
    "point_light": {
        "max_range": 20,       # ~6.7 m
        "angle_spread": 2π,
        "intensity": 1.0,
        "color": (1.0, 0.9, 0.8),
        "heat": 0.0,
        "jitter": 0.0,
        "falloff_fn": "uniform",
    },
    "fire": {
        "max_range": 15,       # ~5 m
        "angle_spread": 2π,
        "intensity": 0.8,
        "color": (1.0, 0.6, 0.2),
        "heat": 1.0,
        "jitter": 0.05,        # ~3° — shadow edges dance
        "falloff_fn": "uniform",
    },
    "flashlight": {
        "max_range": 30,       # 10 m
        "angle_spread": 0.7,   # ~40° cone
        "intensity": 1.5,
        "color": (1.0, 1.0, 0.95),
        "heat": 0.0,
        "jitter": 0.0,
        "falloff_fn": "cosine",  # bright center, fades at edges
    },
    "energy_weapon": {
        "max_range": 40,       # ~13 m
        "angle_spread": 0.14,  # ~8° beam
        "intensity": 3.0,
        "color": (0.3, 0.8, 1.0),
        "heat": 2.0,
        "jitter": 0.0,
        "falloff_fn": "sharp",  # near-uniform within cone, hard cutoff
    },
    "muzzle_flash": {
        "max_range": 12,       # ~4 m
        "ray_count": 64,       # brief, doesn't need full coverage
        "angle_spread": 2π,
        "intensity": 5.0,
        "color": (1.0, 0.9, 0.5),
        "heat": 0.0,
        "jitter": 0.02,
        "falloff_fn": "uniform",
        # duration: 1-2 frames only (handled by caller, not raycaster)
    },
    "emergency_light": {
        "max_range": 12,
        "angle_spread": 2π,
        "intensity": 0.4,
        "color": (1.0, 0.1, 0.05),
        "heat": 0.0,
        "jitter": 0.01,        # slow, subtle flicker
        "falloff_fn": "uniform",
    },
}
```

All values are configurable per instance. Profiles are defaults — any field can be overridden when placing a source.

### Ray count logic

```python
def get_ray_count(source):
    if source.ray_count is not None:
        return source.ray_count   # explicit override
    # Default: enough rays that at max_range, adjacent endpoints are ≤ 1 tile apart
    full_circle_rays = int(math.ceil(2 * math.pi * source.max_range))
    # Scale down for cones (no need for 192 rays in a 40° flashlight)
    fraction = source.angle_spread / (2 * math.pi)
    return max(8, int(math.ceil(full_circle_rays * fraction)))
```

A 40° flashlight at range 30: `ceil(192 * 40/360)` = 22 rays. Cheap.

### Ray marching (DDA tile walk)

The core loop. Each ray marches tile-by-tile from source outward:

```python
def march_ray(source, angle, ray_intensity, gmap, light_map, heat_map):
    """March a single ray, depositing light and heat at each tile."""
    # DDA setup (Digital Differential Analyzer)
    dx = math.cos(angle)
    dy = math.sin(angle)
    x, y = source.x, source.y
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1

    # DDA distances (how far along ray to cross one tile boundary)
    dt_dx = abs(1.0 / dx) if dx != 0 else float('inf')
    dt_dy = abs(1.0 / dy) if dy != 0 else float('inf')

    # Distance to first tile boundary
    t_max_x = (0.5 * step_x + 0.5) * dt_dx  # simplified — depends on sub-tile position
    t_max_y = (0.5 * step_y + 0.5) * dt_dy

    remaining_intensity = ray_intensity
    distance = 0.0

    while distance < source.max_range and remaining_intensity > 0.01:
        # Bounds check
        if not in_bounds(x, y, gmap):
            break

        # --- Deposit at current tile ---
        # Inverse-square falloff with distance (or linear, tunable)
        dist_atten = 1.0 / (1.0 + distance * distance * 0.01)  # tunable
        deposited = remaining_intensity * dist_atten

        light_map[y, x] += deposited
        if source.heat > 0 and gmap.is_wall[y, x]:
            heat_map[y, x] += deposited * source.heat

        # --- Absorption at current tile ---
        # Full block: wall
        if gmap.is_wall[y, x] and distance > 0:
            # Reflection (optional, if bounces remain):
            #   compute reflected angle, call march_ray recursively
            #   with remaining_intensity * material_reflection_coeff
            break

        # Partial block: smoke
        smoke_density = gmap.smoke[y, x]
        remaining_intensity *= (1.0 - smoke_density * SMOKE_ABSORPTION_RATE)

        # Partial block: units (absorbed by players — important for radiation)
        if gmap.obstacles[y, x] and not gmap.is_wall[y, x] and distance > 0:
            # Unit absorbs radiation
            # deposit damage/heat to the unit at this tile
            remaining_intensity *= 0.1  # mostly absorbed, some passes through
            # (or break entirely for full occlusion)

        # --- Step to next tile (DDA) ---
        if t_max_x < t_max_y:
            x += step_x
            distance = t_max_x
            t_max_x += dt_dx
        else:
            y += step_y
            distance = t_max_y
            t_max_y += dt_dy
```

### Casting all rays for a source

```python
def cast_light(source, gmap, light_map, heat_map):
    ray_count = get_ray_count(source)
    base_angles = np.linspace(
        source.angle_center - source.angle_spread / 2,
        source.angle_center + source.angle_spread / 2,
        ray_count, endpoint=False
    )

    # Apply jitter (fire flicker, etc.)
    if source.jitter > 0:
        base_angles += np.random.uniform(-source.jitter, source.jitter, size=ray_count)

    for angle in base_angles:
        # Angular intensity falloff
        offset = angle_diff(angle, source.angle_center)
        normalized_offset = offset / (source.angle_spread / 2 + 1e-6)

        if source.falloff_fn == "uniform":
            angular_intensity = 1.0
        elif source.falloff_fn == "cosine":
            angular_intensity = math.cos(normalized_offset * math.pi / 2)
        elif source.falloff_fn == "sharp":
            angular_intensity = 1.0 if normalized_offset < 0.9 else 0.0

        ray_intensity = source.intensity * angular_intensity
        if ray_intensity > 0.01:
            march_ray(source, angle, ray_intensity, gmap, light_map, heat_map)
```

### Shadow caching (optimization)

Split the computation into two layers to avoid redundant work:

1. **Shadow map** (binary: blocked by geometry or not) — recompute only when:
   - A wall is destroyed or created
   - A door opens or closes
   - The light source moves
   - Fire state changes (new fire starts, fire extinguished)

2. **Attenuation** (continuous: dimmed by smoke) — recompute every frame by walking cached ray paths through current smoke density. Cheap because geometry discovery is already done.

For the prototype: skip caching, recast everything every frame. Optimize later if needed. The cost estimates below suggest it may never be needed in C++.

### Cost estimates

| Scenario | Sources | Rays total | Avg tiles/ray | Tile ops | Estimate (Python) | Estimate (C++) |
|---|---|---|---|---|---|---|
| One flashlight | 1 | 22 | 25 | 550 | <1 ms | ~μs |
| 5 room lights | 5 | 640 | 15 | 9,600 | ~2 ms | ~μs |
| 10 fires burning | 10 | 940 | 12 | 11,300 | ~3 ms | ~μs |
| Worst case (20 sources) | 20 | 2,000 | 20 | 40,000 | ~8 ms | <1 ms |

For reference: the wave equation substeps already do full-grid Laplacians (~7,500 tiles × 4 neighbors × multiple substeps per tick). The raycaster is comparable or cheaper.

### Integration with other systems

- **Fire → raycaster**: each burning tile is a light source with the "fire" profile. Heat emission deposits into `gmap.temperature` on solid tiles hit by rays.
- **Raycaster → temperature**: heat deposited by radiation feeds into the temperature diffusion system (Section 1). Fire heats a distant wall via radiation, that wall conducts heat through metal.
- **Raycaster → stealth**: sample `light_map[unit.y, unit.x]`. Below threshold → unit is in shadow → harder for AI to detect.
- **Raycaster → smoke visuals**: wherever `smoke > 0` and `light_intensity > 0` along a ray path, render a volumetric light shaft (god ray approximation).
- **Energy weapons**: fire the raycaster with an "energy_weapon" profile along the shot vector. Deposits heat and damage at every tile the beam passes through. Walls absorb and heat up. Units take radiation damage.

### Reflection (secondary, implement after core works)

When a ray hits a wall and bounces remain:

```python
# Material reflection coefficient
# MAT_HULL: 0.9 (metal reflects most light)
# MAT_WOOD: 0.3 (absorbs most)
# MAT_GLASS: 0.7
refl_coeff = MATERIAL_REFLECTION[gmap.material[y, x]]
reflected_intensity = remaining_intensity * refl_coeff

if reflected_intensity > 0.05 and bounces_remaining > 0:
    reflected_angle = compute_reflection(angle, wall_normal)
    march_ray(source, reflected_angle, reflected_intensity,
              gmap, light_map, heat_map, bounces=bounces_remaining - 1)
```

Default: 1 bounce max for prototype. Configurable. Metal corridors bounce flashlight beams. Wood absorbs. The wall normal is determined by which face the ray entered from (known from DDA step direction).

---

## 4. Lightning Bolt Effect (Electrical Arc)

### Decision

Electrical arcs triggered by events (damaged electronics, energy weapon impacts, exposed wiring, etc.). Visually computed via **recursive midpoint displacement** — cheap, dramatic, and recalculated each frame for a flickering alive quality.

### Triggers

An arc is spawned by game events, not by a continuous system. Examples:

- Ship electronics taking damage (explosion near a control panel)
- Energy weapon hitting metal (discharge on impact)
- Exposed wiring after wall destruction
- Electrical weapon fired by a unit
- Environmental hazard (damaged power conduit)
- Future: water interaction (bolt hits water → conducts to all units standing in connected water tiles)

### Function signature

```python
def spawn_arc(origin_x, origin_y, energy, radius, gmap):
    """
    Trigger an electrical arc from an origin point.

    Args:
        origin_x, origin_y: tile where the arc starts (damage event location)
        energy: arc intensity (determines damage, brightness, visual thickness)
        radius: max search distance for target (tiles)
        gmap: game map (for material lookups)

    Returns:
        Arc object with path, target, damage info (or None if no valid target)
    """
```

### Target selection — path of least resistance

Search within `radius` for the best conductor. Priority order:

1. **Metal tiles** (hull, machinery) — electricity seeks metal. Nearest metal tile wins.
2. **Water tiles** (future) — good conductor, secondary priority.
3. **Units** — conductive enough to arc to (especially if wearing metal armor).
4. **Random air tile nearby** — fallback for a wild spark with no clear target.

```python
def find_arc_target(origin_x, origin_y, radius, gmap):
    best_target = None
    best_priority = 999
    best_dist = radius + 1

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = origin_x + dx, origin_y + dy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > radius or not in_bounds(nx, ny, gmap):
                continue

            mat = gmap.material[ny, nx]
            if mat == MAT_HULL:
                priority = 0    # metal — best conductor
            elif is_water(nx, ny, gmap):  # future: water system
                priority = 1
            elif has_unit(nx, ny):
                priority = 2
            else:
                continue        # skip non-conductive tiles

            if priority < best_priority or (priority == best_priority and dist < best_dist):
                best_target = (nx, ny)
                best_priority = priority
                best_dist = dist

    # Fallback: random nearby tile (wild spark)
    if best_target is None:
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(2, radius * 0.5)
        best_target = (origin_x + int(dist * math.cos(angle)),
                       origin_y + int(dist * math.sin(angle)))

    return best_target
```

### Bolt path — recursive midpoint displacement

The algorithm that Half-Life used. Simple, fast, looks great:

```python
def generate_bolt_path(x1, y1, x2, y2, depth=5, displacement=None):
    """
    Generate a jagged lightning bolt path between two points.

    Args:
        x1, y1: start point (origin)
        x2, y2: end point (target)
        depth: recursion levels (4-6 is good, more = finer detail)
        displacement: initial max perpendicular offset (default: distance/4)

    Returns:
        List of (x, y) points defining the bolt path.
    """
    if displacement is None:
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        displacement = dist / 4.0

    if depth == 0:
        return [(x1, y1), (x2, y2)]

    # Midpoint
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    # Perpendicular direction
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy) + 1e-6
    perp_x = -dy / length
    perp_y = dx / length

    # Displace midpoint randomly along perpendicular
    offset = random.uniform(-displacement, displacement)
    mx += perp_x * offset
    my += perp_y * offset

    # Recurse on each half with halved displacement
    left = generate_bolt_path(x1, y1, mx, my, depth-1, displacement/2)
    right = generate_bolt_path(mx, my, x2, y2, depth-1, displacement/2)

    return left + right[1:]  # avoid duplicate midpoint
```

At depth=5: produces ~33 points. At depth=6: ~65 points. Negligible cost.

### Bolt flicker

Regenerate the path **each frame** the bolt is visible (2-3 frames). Same endpoints, new random displacements. The bolt appears to crackle and dance. This is the key visual trick — static bolts look dead, regenerated bolts look alive.

```python
class Arc:
    def __init__(self, origin, target, energy, duration_frames=3):
        self.origin = origin
        self.target = target
        self.energy = energy
        self.frames_remaining = duration_frames
        self.path = None  # regenerated each frame

    def update(self):
        self.path = generate_bolt_path(*self.origin, *self.target, depth=5)
        self.frames_remaining -= 1
        return self.frames_remaining > 0

    def render(self, screen):
        # Draw bolt as connected line segments
        # Color: white core, blue-white glow
        # Thickness proportional to energy
        for i in range(len(self.path) - 1):
            p1, p2 = self.path[i], self.path[i+1]
            # bright core (white)
            draw_line(screen, (220, 230, 255), p1, p2, width=1)
            # glow (blue, wider, semi-transparent)
            draw_line(screen, (100, 150, 255), p1, p2, width=3, alpha=0.3)
```

### Damage model

Damage is applied along the bolt path. Any unit whose tile is on or adjacent to the bolt path takes electrical damage:

```python
def apply_arc_damage(arc, units, gmap):
    # Collect tiles the bolt passes through
    bolt_tiles = set()
    for (px, py) in arc.path:
        tx, ty = int(round(px)), int(round(py))
        bolt_tiles.add((tx, ty))
        # Also adjacent tiles (electricity arcs slightly)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            bolt_tiles.add((tx+dx, ty+dy))

    damage = arc.energy * ELECTRICAL_DAMAGE_MULT

    for unit in units:
        if (unit.fx, unit.fy) in bolt_tiles and unit.alive:
            unit.hp -= damage
            # Could also: stun, disable equipment, etc.
```

### Water conduction (future — hook now, implement with fluid system)

When the fluid dynamics system is implemented, add this to `apply_arc_damage`:

```python
    # If bolt endpoint is a water tile, flood-fill connected water
    # and damage all units standing on wet tiles
    tx, ty = arc.target
    if is_water_tile(tx, ty, gmap):
        wet_tiles = flood_fill_water(tx, ty, gmap)  # returns set of (x,y)
        water_damage = arc.energy * WATER_CONDUCT_MULT  # reduced but area-of-effect
        for unit in units:
            if (unit.fx, unit.fy) in wet_tiles and unit.alive:
                unit.hp -= water_damage
```

This hook costs nothing until water exists. When it does, electrical arcs become area-of-effect hazards in flooded rooms — shoot out the aquarium, water floods the corridor, then an electrical fault arcs into the water and everyone standing in it gets hit.

### Integration with other systems

- **Raycaster**: a bolt produces a brief bright flash at the origin (spawn a 1-frame muzzle_flash-style light source at the arc origin for the lighting system to pick up)
- **Fire**: if the bolt passes through flammable material, small chance of ignition (hot plasma)
- **Temperature**: bolt deposits heat at origin and target tiles (feeds into Section 1)
- **Smoke**: bolt passing through smoke could ionize it — purely visual, faint glow along path in smoky areas

### Cost

Generating a bolt path: ~60 point calculations per bolt. Rendering: ~60 line segments. Damage check: iterate units against ~60 tiles. Total: trivially cheap. Multiple simultaneous arcs (e.g. cascading electrical failure) are no problem.
