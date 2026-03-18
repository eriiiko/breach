# Breach — Fluid Sim Design Notes
_Distilled from design sessions, March 2026_

---

## Core Fields

Three scalar fields on the grid:

- `p` — atmosphere/pressure, bulk air pressure field. Equilibrium = 1.0, vacuum = 0.0.
- `wave_p` — shockwave perturbation field, zero-mean. Represents fast transient pressure waves from explosions.
- `rho` — smoke/gas density, a passive scalar field.

The wind field `u = grad(p + wave_p)` couples pressure to gas transport.

---

## Two-Wave Architecture (learned March 18, 2026)

### Why two fields?

We tried unifying wave_p and atmosphere into a single field. It blew up. The root cause:

1. **Fire continuously perturbs atmosphere** (O2 consumption creates gradients)
2. **Diffusion creates gradients** (smoothing between rooms at different pressures)
3. **The wave equation amplifies every gradient** at rate c² = 90,000
4. **Damping must exceed the perturbation rate** — damping=3 was insufficient, damping=30 worked but killed shockwave visuals

With two fields, the fast wave (wave_p) only has energy from explosions and damps cleanly to zero. The slow atmosphere field handles bulk pressure changes without wave amplification.

### The two waves

| Field | Purpose | Damping | Mean | Drives |
|---|---|---|---|---|
| `wave_p` | Explosion shockwaves | Low (3/s) — dramatic, visible | Zero-mean enforced | Smoke advection (fast push) |
| `p` (atmosphere) | Bulk air pressure, decompression | High (30/s) — stable | Tracks real mass | Decompression wind, fire O2 |

**One Laplacian**: computed once per substep, both waves use it. The expensive part (neighbor lookups with Neumann BCs) is shared.

### Explosion physics — shockwave then wind

Real explosions produce two effects in sequence:
1. **Shockwave** (supersonic pressure front) — modeled by `wave_p`. Fast, transient, zero-mean.
2. **Blast wind** (bulk air flow outward, then reverse wind filling the void) — modeled by depositing a small amount of pressure into `p`. Slower, sustained.

The deposit into `p` should be small (a grenade adding ~3 units to a room of ~500 is 0.6% — physical). The shockwave is driven by the Laplacian of the pressure spike, not by the absolute amount deposited.

Alternative to depositing: **displace** air instead (remove from center, add to neighbors). This creates the same Laplacian curvature and thus the same shockwave, without adding net mass.

---

## Boundary Conditions — CRITICAL LESSONS

### Neumann BC at walls (reflection)
- Walls are obstacles. The Laplacian **reflects** at walls: neighbor value = self value.
- **NEVER force atmosphere to a specific value at wall tiles.** This creates a Dirichlet condition that contradicts the Neumann reflection and injects energy into the wave equation continuously.
- Wall tile atmosphere values are meaningless — they're never read by neighbors.
- When a wall is **destroyed**, set its atmosphere to the mean of neighboring non-wall tiles (`_neighbor_mean`). Already implemented.

### Dirichlet BC at vacuum (p = 0)
- Vacuum tiles on the map boundary: `p = 0` enforced every substep.
- Hull walls shield the interior (they're obstacles with Neumann BC).
- When hull is breached, the interior suddenly "sees" the vacuum tile through the gap.
- The wave equation propagates a low-pressure front inward — decompression happens at wave speed.

### What went wrong (March 18 debugging session)
We accidentally applied **Dirichlet BC at wall tiles** (`atmosphere[is_wall] = 1.0` every substep). This created a permanent pressure gradient between fire-consumed air (~0.85) and the wall (forced to 1.0). The wave equation amplified this gradient continuously, causing exponential blowup. The fix: only apply Dirichlet at vacuum, leave walls alone.

---

## Update Schedule

| Update | Equation | Frequency | Notes |
|---|---|---|---|
| Fast wave (wave_p) | `p_tt = c² L(p) - gamma_fast * p_t` | Every wave tick | Shockwaves, zero-mean |
| Slow wave (atmosphere) | `p_tt = c² L(p) - gamma_slow * p_t` | Every wave tick | Decompression, bulk flow |
| Wind field | `u = grad(p + wave_p)` | After wave ticks | Stored, reused by all gas types |
| Gas advection (all types) | `rho_t + u . grad(rho) = 0` | Every wave tick | Rides on wind |
| Smoke/gas diffusion | `rho_t = D L(rho)` | Once per game tick (slow) | Diffusion is slow in still air |

**Key principle:** Wind field computed once, stored, reused for ALL gas types. Do not recompute gradients per gas type.

---

## Gas Types (planned)

All are passive scalars advected by the same wind field:

- Smoke
- Tear gas
- Poison gas
- Burnable gas

Cost scales cheaply with number of gas types (one advection pass per type, shared wind).

---

## Decompression / Breach

### How it works
When hull is breached, the destroyed tile becomes air. The vacuum tile behind it has Dirichlet BC (`p = 0`). The wave equation propagates a low-pressure front inward. The gradient `grad(p)` near the breach points toward the hole. All gas types get advected out via normal advection.

**Breach size controls drain rate naturally** — more cells destroyed = stronger sink.

When breach is sealed (if we add that mechanic): remove the vacuum exposure, winds stop.

### Diffusion on atmosphere — possibly unnecessary
With the slow damped wave handling bulk air movement, atmospheric diffusion may be redundant. The wave equation already spreads pressure via the Laplacian. Diffusion would only add slow molecular spread on top — which is negligible compared to wave-driven flow. Consider removing atmosphere diffusion entirely and letting the damped wave do all the work.

---

## Fire Rethink (March 18, 2026)

### Problem
Fire currently consumes atmosphere (`atmosphere -= o2_consumption * dt * fire`). But atmosphere is pressure, not O2. This creates continuous gradients that destabilize the wave equation on the unified field. Even with two fields, it's conceptually wrong.

### New fire design
- **Remove fire consuming atmosphere entirely.** Fire should not modify the pressure field.
- **Fire dies when atmosphere is too low** — low atmosphere is the proxy for "no O2". Threshold check only, no continuous consumption.
- **Wind logic stays**: wind speed from `grad(p)` modulates fire intensity. Weak wind + weak fire = feeds. Strong wind + weak fire = blown out. Strong wind + strong fire = burns hotter.
- **Fire only modifies**: its own intensity field, smoke emission, and wall HP (damage).
- **Negative atmosphere from wave equation**: not a problem physically (rarefaction). For fire: treat `atmosphere < threshold` as "no O2", fire goes out. Simple clamp on the check, not on the field.

### Why this is better
1. Fire no longer injects energy into the wave equation
2. Atmosphere field is only modified by: wave equation, diffusion (if kept), vacuum drain, and explosion deposits
3. Cleaner separation of concerns: fire reads atmosphere, never writes it

---

## Zero-Mean Constraint

Applies to `wave_p` only (the shockwave field). Enforced each substep. This prevents DC drift from numerical error in explicit Euler.

Does NOT apply to atmosphere — atmosphere tracks real mass which changes via vacuum drain and explosion deposits.

---

## TODO — Next Session

1. **Merge useful develop changes into main**: wind field storage, proper Neumann BCs, no atmosphere clamping
2. **Keep two fields**: wave_p for shockwaves (damping=3, zero-mean), atmosphere for bulk (damping=30)
3. **Add slow damped wave on atmosphere** for decompression
4. **Smoke advection uses stored wind field** (already done in develop branch)
5. **Test hull breach decompression**
6. **Smoke update frequency**: currently once per tick — may need more frequent updates to see wind effects
7. **Extensive debug system**: log all states in pickles for replay analysis

---

## Equations Reference

**Wave equation (pressure perturbation):**
$$f_{tt} = c^2 \nabla^2 f - \gamma f_t$$

**Laplacian (5-point stencil):**
$$\nabla^2 f \approx f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4f_{i,j}$$

**Gradient (wind):**
$$\mathbf{u} = \nabla f \approx \left(\frac{f_{i+1,j}-f_{i-1,j}}{2},\ \frac{f_{i,j+1}-f_{i,j-1}}{2}\right)$$

**Advection-diffusion (gas):**
$$\frac{\partial \rho}{\partial t} = -\mathbf{u} \cdot \nabla \rho + D\nabla^2\rho$$

**CFL stability condition:**
$$dt \leq \frac{0.707 \cdot dx}{c}$$

**Damping stability:**
$$\gamma \cdot dt < 2$$

**Energy balance (unified field):**
Damping must exceed the rate of gradient injection by fire/diffusion. With c=300, damping >= 30 required for stability when fire is active.
