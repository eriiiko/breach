# Breach — Fluid Sim Design Notes
_Distilled from design session, March 2026_

---

## Core Fields

Two separate scalar fields on the grid:

- `f` — atmosphere/pressure, governed by the **wave equation**
- `ρ` — smoke/gas density, a passive scalar field

These are distinct. The wind field `∇f` is the one-way coupling: pressure drives gas, not the other way around.

---

## Update Schedule

| Update | Equation | Frequency | Notes |
|---|---|---|---|
| Wave equation | `f_tt = c² ∇²f` | Every wave tick (~27.8ms) | Most expensive — full grid Laplacian |
| Wind field | `u = ∇f` | Every wave tick | Free — reuses neighbor samples from wave solve |
| Gas advection (all types) | `ρ_t + u·∇ρ = 0` | Every wave tick | Rides on wind; same CFL stability requirement |
| Smoke/gas diffusion | `ρ_t = D∇²ρ` | Once per game turn (slow) | Physically correct — diffusion is slow in still air |

**Key principle:** Once wind is computed, it is stored and reused for ALL gas types that tick. Do not recompute `∇f` per gas type.

---

## Gas Types (planned)

All are passive scalars advected by the same wind field `u = ∇f`:

- Smoke
- Tear gas
- Poison gas
- Burnable gas

Since they all share the same wind field, advection cost scales cheaply with number of gas types.

### Sparse matrix note (open question)
Gas fields are sparse — they typically occupy only a fraction of the map. With many simultaneous sources this may still warrant sparse matrix representation for the diffusion step. Defer until profiling shows it's needed.

---

## Decompression / Breach — KNOWN BUG & FIX

### The bug
Decompression is currently modeled via diffusion. This is wrong. Diffusion is slow molecular spread in the absence of bulk flow. It will never produce violent decompression winds no matter how high `D` is set.

### What decompression actually is
Bulk mass flow driven by a pressure gradient. Air exits at speeds up to ~343 m/s (speed of sound). This is advection, not diffusion.

### The fix — Dirichlet boundary condition on breach cells

When a breach opens, **force that cell to vacuum every tick:**

```
f[breach_cell] = 0   // enforced every wave tick
```

This is a **Dirichlet boundary condition**. The wave equation then does the rest:
- Low-pressure front propagates inward from the breach
- `∇f` everywhere near the breach points strongly toward the hole
- All gas types get violently advected out via the normal advection step
- Room drains at wave speed, not diffusion speed

**Breach size controls drain rate naturally** — more cells set to `f = 0` → stronger sink. No tuning needed.

When breach is sealed → remove the BC → winds stop immediately.

---

## Zero-Mean Constraint — Open Question

The wave equation is currently kept zero-mean each timestep. This may need revisiting:

- With vacuum BCs (Dirichlet `f = 0` at breach), mass is actively leaving the system
- Zero-mean may no longer hold naturally
- May need to **track net flux escaping at breach boundaries** rather than enforcing zero-mean globally
- This is a non-trivial change — defer to a dedicated session

---

## Summary: What to Change Tonight

1. **Detect breach events** → mark those cells as Dirichlet BC (`f = 0`)
2. **Enforce BC every wave tick** — vacuum doesn't fill up
3. **Remove or reduce reliance on diffusion for decompression** — diffusion rate should be slow (physically realistic), not cranked up to compensate for missing advection
4. Wind field computed once per tick, stored, reused for all gas advection
5. Zero-mean question — leave for later

---

## Equations Reference

**Wave equation (pressure):**
$$f_{tt} = c^2 \nabla^2 f$$

**Laplacian (5-point stencil):**
$$\nabla^2 f \approx f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4f_{i,j}$$

**Gradient (wind):**
$$\mathbf{u} = \nabla f \approx \left(\frac{f_{i+1,j}-f_{i-1,j}}{2},\ \frac{f_{i,j+1}-f_{i,j-1}}{2}\right)$$

**Advection-diffusion (gas):**
$$\frac{\partial \rho}{\partial t} = -\mathbf{u} \cdot \nabla \rho + D\nabla^2\rho$$

**CFL stability condition (both wave and advection):**
$$dt \leq \frac{dx}{c}$$
