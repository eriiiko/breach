# Map Class & Physics Systems Design

## Session Summary

This session established the data architecture for the game map and the core physics simulation systems: explosion propagation (wave equation), atmospheric decompression (diffusion equation), and smoke dynamics (diffusion + advection). The key insight is that all systems share one grid, one wall matrix, and one Laplacian function — emergent complexity arises from layering simple fields.

---

## 1. Map Architecture

### Decision: Tile Objects + Cached Numpy Arrays

The map is a 2D grid of `Tile` objects (rich data, easy to extend) with cached numpy arrays derived from tile state (fast computation for physics).

**Rejected alternative:** Pure tensor/matrix encoding. Becomes brittle when adding new properties — every consumer of the tensor needs updating. Hard to store mixed state (type info + changing values like HP and fire intensity).

### Material System (Data-Driven)

Materials are defined in a single lookup table. Adding a new material = one line, no code changes:

```python
MATERIAL_PROPERTIES = {
    Material.NONE:      {"max_hp": 0,   "flammable": False, "blocks_light": False, "blocks_gas": False, "blast_resistance": 0.0},
    Material.STEEL:     {"max_hp": 200, "flammable": False, "blocks_light": True,  "blocks_gas": True,  "blast_resistance": 0.8},
    Material.WOOD:      {"max_hp": 60,  "flammable": True,  "blocks_light": True,  "blocks_gas": True,  "blast_resistance": 0.2},
    Material.GLASS:     {"max_hp": 15,  "flammable": False, "blocks_light": False, "blocks_gas": True,  "blast_resistance": 0.0},
    Material.HULL:      {"max_hp": 300, "flammable": False, "blocks_light": True,  "blocks_gas": True,  "blast_resistance": 0.9},
}
```

### Tile Class

```python
class Tile:
    material: Material
    hp: int
    max_hp: int
    flammable: bool
    blocks_light: bool
    blocks_gas: bool
    blast_resistance: float
    on_fire: bool
    fire_intensity: float
    temperature: float
    entities: list
    liquid_type: Liquid
    liquid_depth: float
```

### Cached Arrays (GameMap)

Rebuilt when tiles change via `on_tile_changed(x, y)`:

- `_light_block` — 1 where light is blocked (for shadowcasting)
- `_gas_block` — 1 where gas can't pass (for pressure/smoke)
- `_walkable` — 1 where entities can move
- `_flammable` — 1 where fire can spread

### Level Editor Implication

Material-based approach maps directly to painting: select material brush, paint tiles, all properties auto-derive from the material table. Preview layers (light map, gas map, flammable map) toggle on/off for verification.

---

## 2. Shared Infrastructure: Laplacian with Wall Boundaries

All physics systems use the same discrete Laplacian with Neumann boundary conditions (rigid wall reflection):

```python
def compute_laplacian_with_walls(p, gas_block):
    up    = np.roll(p, 1, axis=0)
    down  = np.roll(p, -1, axis=0)
    left  = np.roll(p, 1, axis=1)
    right = np.roll(p, -1, axis=1)

    wall = gas_block == 1
    wall_up    = np.roll(wall, 1, axis=0)
    wall_down  = np.roll(wall, -1, axis=0)
    wall_left  = np.roll(wall, 1, axis=1)
    wall_right = np.roll(wall, -1, axis=1)

    # Neumann BC: mirror center value where neighbor is wall
    up    = np.where(wall_up, p, up)
    down  = np.where(wall_down, p, down)
    left  = np.where(wall_left, p, left)
    right = np.where(wall_right, p, right)

    return up + down + left + right - 4 * p
```

This single function gives all systems: reflection off walls, diffraction through doorways, channeling through corridors.

### Per-Material Reflection Coefficients

Walls can partially absorb pressure waves. No angle calculation needed — the geometry handles it automatically:

```python
MATERIAL_REFLECTION = {
    Material.HULL:  0.95,   # near-perfect reflector
    Material.STEEL: 0.90,
    Material.WOOD:  0.50,   # absorbs significantly
    Material.GLASS: 0.70,
}
```

At wall boundaries, instead of fully mirroring: `reflected_value = ref_coeff * p_center`

---

## 3. Explosion Propagation (Wave Equation)

### Physics

The 2D wave equation: ∂²p/∂t² = c²∇²p

Discretized as: `p_next = 2 * p_now - p_prev + r * laplacian`

Where `r = (c * dt / dx)²` must be ≤ 0.5 for stability (CFL condition).

### Parameters

- `dx = 1.0 m` (1 tile = 1 meter)
- `c = 343 m/s` (speed of sound)
- `dt = 0.001 s`
- `r = 0.1176` (stable)
- ~150 steps for wave to cross a 50-tile ship

### Dissipation

**Decision: No artificial damping constant.** Energy dissipates naturally via 1/√r spreading (pressure amplitude) in 2D. The wave equation produces this automatically.

Instead, use a **max iterations cutoff** based on ship geometry:

```python
max_steps = int((max(width, height) * NUM_TRAVERSALS) / (c * dt))
```

With `NUM_TRAVERSALS = 3-4`, all meaningful reflections are captured. Remaining energy is physically negligible.

### Damage Model

Damage is proportional to the **peak pressure gradient magnitude** experienced at each tile:

```python
grad_x = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dx)
grad_y = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dx)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
```

Track `peak_gradient = np.maximum(peak_gradient, grad)` each sub-step. Apply as damage after simulation completes.

### Wall Destruction Mid-Simulation

When accumulated gradient exceeds wall HP, flip `gas_block[y, x] = 0` during the simulation. The wave then pours through the breach, losing energy in the process. Hull breaches from explosions emerge naturally.

### Emergent Behaviors (Free from the Math)

- Corridor channeling (walls reflect inward, pressure amplifies)
- Blast shadows (hiding behind steel works — low gradient on far side)
- Sealed room devastation (reflections compound)
- Grenade near hull breach = harmless (energy vents to vacuum)
- Diffraction through doorways

### Vacuum Boundary

Vacuum tiles: `p = 0`, reflection coefficient = 0. Perfect absorber — blast energy exits the ship.

---

## 4. Atmospheric Decompression (Diffusion Equation)

### Separate from Explosions

Explosions are transient wave events (milliseconds). Decompression is continuous bulk airflow (seconds/turns). Different timescales, different equations, same grid.

### Physics

The diffusion equation: ∂p/∂t = D∇²p (first order — simpler than wave equation)

```python
def atmosphere_step(atm, gas_block, dt_atm, diffusion_rate):
    lap = compute_laplacian_with_walls(atm, gas_block)
    atm_next = atm + diffusion_rate * dt_atm * lap
    atm_next = np.where(is_vacuum, 0.0, atm_next)  # vacuum stays at 0
    return atm_next
```

### Field

```python
atmosphere = np.where(is_vacuum, 0.0, 1.0)  # persistent game state, 0-1 atm
```

Evolves every turn. Interior starts at 1.0 atm, vacuum fixed at 0.0.

### Gameplay Effects

- Suffocation damage below 0.3 atm
- Rapid incapacitation below 0.1 atm
- **Suction force**: entities pulled along atmosphere gradient toward breaches
- Sealed bulkheads protect compartments
- Opening a door between pressurized and depressurized rooms: pressure equalizes through the doorway

---

## 5. Smoke (Diffusion + Advection)

### Separate field, rides on atmosphere

Smoke is a substance *in* the air, not the air itself. Own density field, carried by airflow.

```python
smoke = np.zeros((height, width))  # 0.0 = clear, 1.0 = opaque
```

### Two forces

1. **Diffusion** — smoke spreads on its own (slow, same Laplacian)
2. **Advection** — smoke carried by wind (atmosphere gradient)

```python
def smoke_step(smoke, atmosphere, gas_block, dt):
    lap = compute_laplacian_with_walls(smoke, gas_block)
    smoke_next = smoke + SMOKE_DIFFUSION * dt * lap

    # Atmosphere gradient = wind direction
    grad_y = (np.roll(atmosphere, -1, axis=0) - np.roll(atmosphere, 1, axis=0)) / 2
    grad_x = (np.roll(atmosphere, -1, axis=1) - np.roll(atmosphere, 1, axis=1)) / 2

    # Advect smoke along airflow
    smoke_next -= ADVECTION_RATE * dt * (
        grad_x * (np.roll(smoke, -1, axis=1) - np.roll(smoke, 1, axis=1)) / 2 +
        grad_y * (np.roll(smoke, -1, axis=0) - np.roll(smoke, 1, axis=0)) / 2
    )

    smoke_next *= 0.995  # slow settling/thinning
    return np.clip(smoke_next, 0, 1)
```

### Interactions

- Fire produces smoke (source term at burning tiles)
- Hull breach sucks smoke out (atmosphere gradient pulls it)
- Sealed room fills up dense
- Smoke blocks vision: `smoke > THRESHOLD` → feeds into `_light_block` cache

---

## 6. System Summary

| System | Field | Equation | Timescale | Persistent? |
|--------|-------|----------|-----------|-------------|
| Explosion | `blast` | Wave (2nd order) | Milliseconds | No — spike, run, collect damage, reset |
| Atmosphere | `atmosphere` | Diffusion (1st order) | Seconds/turns | Yes — evolves all match |
| Smoke | `smoke` | Diffusion + advection | Seconds/turns | Yes — evolves all match |

All three share: `compute_laplacian_with_walls()`, `gas_block` matrix, same tile grid.

### Emergent Cross-System Interactions

- Explosion breaks hull → atmosphere vents → smoke gets sucked out
- Fire consumes oxygen (lowers local atmosphere) → decompression starves fires
- Grenade in sealed room = devastating; near breach = harmless
- Deliberate hull breach to vent fire or smoke = valid tactic
- Opening doors connects pressure regions — wind pulls smoke and entities through

---

## 7. Next Systems to Design

- **Fire propagation** — uses `_flammable` cache, consumes oxygen from atmosphere field
- **Light / Vision (shadowcasting)** — uses `_light_block` cache (affected by smoke)
- **Liquid dynamics** — blood, water, fuel on the same grid pattern
