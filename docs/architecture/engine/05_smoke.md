# Smoke

**Depends on:** [Grid & Coordinates](01_grid_and_coordinates.md) · [State & Ownership](02_state_and_ownership.md) · [Atmosphere & Pressure](04_atmosphere_and_pressure.md)

> Note: the dependency chapters are listed by role — the grid/coordinate model, the
> array-and-table state contract (ch.01), and the atmosphere solver that produces the wind
> field smoke rides on. Where a numbered file does not yet exist, the dependency is on the
> system it names.

---

## 1. What smoke is

Smoke is a single scalar density field, `smoke`, one float per tile in `[0, 1]` — `0` is clear
air, `1` is fully opaque. It is a passive substance *in* the air, not the air itself: it has no
mass of its own and exerts no pressure. It is carried by the airflow the atmosphere solver
already computes, spreads slowly on its own, and dims any ray that crosses it.

Smoke is its own field rather than a property of the atmosphere field because the two evolve on
different physics. Air pressure equalises (diffusion, with acoustic shockwaves on top); smoke is
a tracer that *rides* the resulting wind without changing it. Keeping them separate means the
atmosphere solver never needs to know smoke exists, and smoke gets transport for free from a
field it does not own.

Smoke matters to gameplay through three couplings, all of which fall out of existing systems
rather than special-case code:

- **Vision and rays.** Smoke attenuates light per channel as a ray marches through it, casts soft
  shadows, and — because the absorbed light is captured rather than discarded — produces
  volumetric god-ray shafts. This is the smoke↔ray-engine seam (ch.03), detailed in §5.
- **Fire.** Burning tiles emit smoke into adjacent air; explosions dump a cloud. Fire is the
  primary source (§4).
- **Tactics.** A breach sucks smoke out toward vacuum; a sealed compartment fills with it; a
  grenade clears a pocket then refills it. None of this is scripted — it is the wind field acting
  on a tracer.

The field lives on `GameMap` and is reached as `gmap.smoke`, consistent with the array-and-table
state contract (ch.01): no tile-objects, no per-tile smoke metadata, just one numerical array
that every system reads or writes by name.

---

## 2. The model: diffusion + advection

Smoke evolves by two forces applied every atmosphere substep. Both use fields the atmosphere
solver has already produced this substep — there is no redundant gradient computation.

```
actual_dt   = dt * dt_scale                                  # time amplification
wind_sq     = wind_x² + wind_y²                              # per tile
D_eff       = d_smoke * (1 + wind_diffusion_scale * wind_sq) # turbulent mixing
smoke      += D_eff * actual_dt * laplacian(smoke)           # diffusion
smoke      -= advection_rate * actual_dt * (wind · grad smoke)   # wind transport
clamp smoke to [0, 1]; zero on walls and vacuum
```

**Diffusion** is the shared 4-neighbour Laplacian with Neumann boundary conditions (the same
operator the atmosphere and wave solvers use): where a neighbour is an obstacle, its value is
mirrored from the centre, so smoke reflects off walls, diffracts through doorways, and channels
down corridors with no extra logic. The base coefficient `d_smoke` is deliberately low so calm
smoke holds its shape instead of dissolving into uniform grey.

**Advection** transports smoke along the wind field. The wind is `−grad(atmosphere + wave_p)`,
precomputed by the atmosphere solver into `gmap.wind_x` / `gmap.wind_y` every substep (ch.06).
The advection term is the gradient-dot form `wind · grad(smoke)`: smoke flows down the pressure
gradient, away from overpressure and toward breaches. Using the precomputed wind is what lets
smoke ride a shockwave — the same `wave_p` that drives the blast also drives the smoke in front
of it.

### Why wind-dependent diffusion

The diffusion coefficient scales with the **square** of local wind magnitude
(`wind_x² + wind_y²`), not a constant. This is the key stability-and-aesthetics choice.

A plain explicit advection term has no hard CFL limit but oscillates when `advection_rate ·
actual_dt · |wind| · |grad smoke|` approaches 1 — exactly the regime near breaches and
explosions, where wind is strong. Left alone, those high-wind boundaries develop checkerboard
artifacts. Coupling diffusion to wind smooths precisely the high-frequency modes that advection
would otherwise destabilise, *where* they appear, while leaving calm interiors almost untouched.

The quadratic (rather than linear) response sharpens this split: in still air `wind_sq ≈ 0` so
`D_eff ≈ d_smoke` and smoke keeps crisp, interesting structure; in a gale `wind_sq` is large so
`D_eff` rises steeply and the turbulent region mixes out before it can ring. Physically this
reads as turbulent mixing — fast-moving air disperses smoke faster — so the stabiliser and the
visual goal point the same way.

### Why smoke is interleaved with atmosphere, not stepped once per tick

Smoke is advanced **every atmosphere substep** (~50× per game tick at the wave CFL), inside the
same loop as the atmosphere solver, immediately after it. It does not run once per tick at the
full ~83 ms `dt`.

The reason is the shockwave. A blast crosses the map in a few milliseconds; if smoke only saw the
final post-tick wind it would teleport rather than billow. Interleaving means each tiny wind
update immediately pushes the smoke a tiny step, so smoke visibly rolls ahead of the pressure
front in real time. `dt_scale` (default `3.0`) is a deliberate, non-physical amplification of the
smoke timestep: it makes smoke react faster and more dramatically than a literal reading of the
wind would give, which reads better on screen without affecting the atmosphere it rides on.

The orchestration (per tick) is therefore:

```
dt = AtmosphereSolver.max_dt()        # wave CFL, ~1.67 ms
n  = ceil(sim_time / dt)              # ~50 substeps
for each substep:
    AtmosphereSolver.step(dt)        # wave + diffusion + BCs + wind field
    SmokeDynamics.step(dt * dt_scale) # diffusion + advection on the fresh wind
after the loop:
    FireSimulation.step(sim_time)    # emits smoke (single full-tick step)
```

Fire runs once per tick after the substep loop because it only needs the final atmospheric state;
its smoke emission therefore lands as a per-tick deposit, not a per-substep one.

### Boundary handling

Smoke's boundary is the **gas-flow** boundary, not the light-occlusion one: smoke is stopped by what
is impermeable to gas, which is *not* the same set as what blocks light (a grill passes both light
and smoke; glass blocks smoke yet passes light). Today the solver reads the boolean `obstacles` mask
(walls + units) — the interim form — and per the coefficient model (ch.03) this becomes a per-cell
**permeability**: walls `0` (sealed), units/grills partial. Walkability is a different predicate
again and is irrelevant to smoke.

Vacuum tiles are a hard sink: smoke on any `is_vacuum` tile is zeroed each step (interior smoke
clamped to `[0, 1]`), and advection is skipped on impermeable and vacuum tiles (they would read
garbage gradients). Zeroing vacuum is what makes a breach drain a room — smoke advects toward the
breach and is deleted there, exactly as venting to space should look.

**Lingering-smoke fix (owed).** With the current vacuum-*relaxation* drain, the wind that carries
smoke out dies as interior pressure approaches zero (the gradient vanishes), so a ship breached to
vacuum keeps a stubborn haze long after the air is gone — it looks wrong. The fix is the atmosphere
chapter's **face-flux drain** (ch.04): outflow to vacuum sustains a face velocity even at near-zero
pressure, so the wind keeps dragging smoke through the breach until the room is clear. (`dt_scale`
and `advection_rate` already let smoke move faster than a literal wind reading for feel; that is
tuning, not the root fix — face-flux is.)

### Parameters

All four parameters live on the C++ `SmokeDynamics` instance and are bound from `config.toml`
at init (`PhysicsRunner.__init__`). None are hardcoded in the solver.

| Parameter | Config key | Default | Role |
|---|---|---|---|
| `d_smoke` | `physics.d_smoke` | `0.1` | Base diffusion (low → smoke holds shape) |
| `advection_rate` | `physics.advection_rate` | `100.0` | Wind transport strength |
| `dt_scale` | `physics.smoke_dt_scale` | `3.0` | Smoke timestep multiplier (visual amplification) |
| `wind_diffusion_scale` | `physics.wind_diffusion_scale` | `50.0` | Turbulent-mixing strength: `D = d_smoke·(1 + scale·\|wind\|²)` |

---

## 3. Determinism and numerics

Smoke is a render-and-tactics field that never crosses a hard gameplay threshold of its own —
nothing ignites or dies *because* `smoke` ticked past a specific value — so it stays `float32`.
This is consistent with the project rule: fixed-point integers only where a value crosses a
discrete threshold into sim state (heat, temperature); float where the quantity is continuous and
perceptual (ch.04). Smoke's only sim-side consequence is attenuating rays, which is itself a
continuous multiply.

Determinism for AI rollouts is preserved at the *source*, not in the solver: the explosion-smoke
noise is drawn from the Simulation facade's seeded `numpy.random.Generator`, never the
process-global RNG (§4). The diffusion/advection update is a deterministic bulk array operation.

On CUDA, smoke is one more scalar field on the same grid; the diffusion stencil is the textbook
2D Laplacian kernel and advection becomes a semi-Lagrangian back-trace using the GPU's hardware
texture interpolation (the Stable-Fluids pattern). No per-tile atomics are involved — each cell
writes only its own value — so smoke needs no fixed-point treatment to remain reproducible across
the CPU→GPU migration.

---

## 4. Sources

Smoke is added by two systems; the solver itself only transports and dissipates.

**Fire (continuous).** Each burning tile emits into its 4-connected air neighbours every fire
step: `smoke[neighbour] += smoke_emission · dt · fire_intensity`. This is part of the
`FireSimulation` step (ch.08), so a sustained fire produces a sustained plume that the wind then
carries. A fire near a breach starves *and* its smoke is sucked out — the emergent chain the
design aims for, with no code linking the two.

**Explosions (impulse).** A detonation does two smoke things, in two places:

- `apply_explosion` **clears** smoke in the inner 40 % of the blast radius — the fireball
  punches a hole in any existing cloud.
- `add_explosion_smoke` **deposits** a fresh disc: `base = 0.8·(1 − dist/radius)`, multiplied by
  a per-tile random factor in `[0.4, 1.0]` drawn from the seeded generator, accumulated into the
  existing density and clamped to `1`.

**Known issue — explosion smoke noise is too subtle.** In practice the per-tile `[0.4, 1.0]`
multiplier does not produce visible texture: tiles near the centre saturate at `1.0` regardless
of the noise, and the low base diffusion smooths the disc edge within a few substeps. The cloud
reads as a flat blob rather than a ragged, churning mass. The fix is more dramatic initial
structure — missing patches (some tiles seeded near zero), larger-scale spatial noise, or a
density falloff that does not saturate — so that advection has high-frequency structure to grab
and carry. This is a tuning/authoring change to the source, not a change to the transport model.

---

## 5. Smoke and the ray engine

Smoke is the ray engine's one **dynamic** per-tile attenuator. The ray march (ch.03) multiplies
two attenuations as it crosses each tile: the **static** per-channel material attenuation from
the table (`light_atten`, a structural-change cache), and the **live** smoke attenuation derived
from the current `smoke` density. Smoke generalises the same multiplicative mechanism the material
table uses — the only difference is that its multiplier comes from a field that changes every tick
rather than a constant indexed by material id.

Per tile, per channel, the directional march does:

```
absorb_frac = smoke_density · smoke_absorption        # fraction this tile removes
smoke_glow[tile] += deposited_light · absorb_frac     # capture the absorbed light (RGB)
remaining        *= (1 − absorb_frac)                 # dim the surviving ray
```

Two things follow from capturing rather than discarding the absorbed light:

**God-rays are energy-conserving and coloured.** The light a smoke tile removes from the ray is
added to `smoke_glow`, an RGB render buffer (ch.05). Because the deposit is per channel, a red
beam casts a red shaft and a white beam a white one — the glow is whatever colour the smoke ate.
This *is* the god-ray mechanism: there is no separate render-time shaft hack and no double
counting. `smoke_glow` supersedes the older flat surface-tint (`light_modulation`) path; the
render side draws it additively as the volumetric shaft. (Smoke attenuation in the *light* march
uses a scalar `smoke_absorption`; the per-channel split is in the colour the deposit and the
survivor carry, not in three separate smoke coefficients.)

**Smoke shadows are free.** Because each smoke tile dims the ray that passes through it, tiles
behind a dense cloud receive less light automatically — soft shadowing with no shadow-specific
code, the same way an opaque wall produces a hard shadow by killing the ray entirely.

Both the legacy scalar light path (`march_ray` → `light_map`, used by `update_from_fire`) and the
current RGB directional path (`march_ray_directional` → `light_rgb` / `heat` / `smoke_glow`) read
the same `smoke` field; only the RGB path performs the god-ray deposit.

---

## 6. Forward design ideas

These are routed, agreed directions layered on top of the shipped model — not yet built. They are
recorded here so the canon model and its extension points stay in one place.

**Normal-mapped smoke (render).** Smoke density is one value per physics tile, but the *visual*
smoke pixels within a tile sit at render resolution. A per-pixel "smoke normal" sub-texture would
let smoke not just attenuate light but catch directional highlights from it, using the same
per-pixel dot-product the renderer already does for solid sprites (the light direction is in
`gmap.light_dir`). The result is internal shading inside the volume — wisps and eddies reading as
structure instead of flat fog. Cheap because the renderer already samples light direction per
pixel; the only new cost is authoring or generating the normal texture. (See ch.05.)

**Multi-gas system (colour, poison, teargas, fuel).** Smoke generalises from one scalar field to a
small **set of gas density fields** `(h, w, N)` — one per gas type — sharing the *same* diffusion +
advection solver (they all ride the same wind; on CUDA it is one batched stencil). Normal smoke
becomes gas type 0. The planned set is **normal smoke · poison · tear gas · flammable fuel gas**;
because a gas type is a data row, white/black smoke or any other variant are free additions later —
just more rows, not more system.

A gas type is **data-driven, exactly like a material** — a `[gases.*]` config table, one row per gas:

- **Optical signature = colour.** Each gas carries a *per-channel* attenuation triple (the
  per-channel `smoke_absorption` the shipped model lacks — see Gaps). That single coefficient is both
  how the gas *looks* and how it *tints light* through it: green poison absorbs red+blue and passes
  green, so it reads green *and* greens the light behind it. **Mixing falls out for free** as the
  density-weighted sum of signatures — poison over fuel blends to a murky olive in both the look and
  the light-tint, through the same per-channel attenuation combine `dyn_light_atten` already does. No
  blend code.
- **Effect** — poison = damage-over-time to a unit in the cell; teargas = slow / blind. (A unit-side
  reading of the gas fields; mechanics chapter.)
- **Flammability** — fuel ignites where `fuel > threshold` with oxygen/heat present, spawning fire. A
  **flamethrower** is then just: emit the fuel gas in a cone (directed injection — the
  momentum-at-nozzle idea from atmosphere) and ignite it. Fuel + the advection model + fire, no new
  mechanic.

This unifies what were separate forward ideas (a fuel field, a teargas field) into one system and
reuses 100 % of the smoke transport plus the per-channel attenuation machinery. v1 keeps the gases
independent — they coexist and blend *visually*, each with its own effect; chemical interaction
between gases is a later layer. Cost is ≈ N× the (cheap, batchable) smoke solver.

---

## Implementation status

Audited against `cpp/src/smoke_dynamics.{h,cpp}`, `src/simulation/physics_runner.py`,
`src/simulation/physics.py`, `src/simulation/gamemap.py`, `cpp/src/raycaster.cpp`,
`cpp/src/fire_simulation.cpp`, and `config.toml`.

**Built and shipped:**

- **Transport solver** — `SmokeDynamics::step` implements wind-dependent diffusion (D scales with
  `wind_diffusion_scale · |wind|²`), `grad_p · grad_smoke` advection by the precomputed wind
  field, `dt_scale` amplification, clamp to `[0,1]`, and wall/vacuum zeroing — exactly as
  described in §2. Built in C++.
- **Atmosphere interleaving** — `PhysicsRunner.step` runs `smoke.step(dt · dt_scale)` inside the
  per-substep loop, immediately after `atmos.step`, reading the freshly-computed
  `wind_x`/`wind_y`. Matches §2.
- **Parameters from config** — all four (`d_smoke`, `advection_rate`, `smoke_dt_scale`,
  `wind_diffusion_scale`) are bound from `config.toml` in `PhysicsRunner.__init__`. Defaults in
  the doc match `config.toml` (`0.1 / 100.0 / 3.0 / 50.0`). Note the *C++ class defaults*
  (`d_smoke=0.4`, `advection_rate=25.0`, `wind_diffusion_scale=0.0`) differ but are always
  overwritten at init, so config wins.
- **Sources** — fire smoke emission (`fire_simulation.cpp`, into 4-connected air neighbours);
  explosion smoke clear (inner 40 %, `physics.apply_explosion`) and noisy disc deposit
  (`physics.add_explosion_smoke`) via the seeded RNG. Matches §4.
- **Ray-engine coupling** — `raycaster.cpp` `march_ray_directional` applies live smoke attenuation
  per channel after static material attenuation, and deposits the absorbed light into the RGB
  `smoke_glow` buffer (god-rays), superseding `light_modulation`. The legacy scalar path
  (`march_ray`) also attenuates by smoke. Matches §5. `smoke` and `smoke_glow` are allocated on
  `GameMap` and written in place. `smoke` stays `float32` (§3).

**Designed but not built:**

- **Permeability boundary + lingering-smoke fix** — smoke's boundary is still the boolean
  `obstacles`/occlusion mask; per ch.03/ch.04 it becomes a gas **permeability** (units/grills
  partial, walls sealed), and the lingering-haze-on-vacuum artifact is fixed by the atmosphere
  **face-flux** drain. Both owed.
- **Normal-mapped smoke** (§6) — render-side idea only; no smoke-normal texture or shader path.
- **Multi-gas system** (§6) — smoke is a single scalar field today; the N-field gas set
  (poison / teargas / fuel, a data-driven `[gases.*]` table, per-channel colour/attenuation, and
  density-weighted mixing) is not built. Flamethrower (fuel + ignition) follows from it.
- **CUDA path** (§3) — semi-Lagrangian GPU advection is planned; the current solver is CPU C++.

**Gaps / known issues:**

- **Explosion-smoke noise too subtle** (§4) — confirmed in code: the `[0.4, 1.0]` per-tile
  multiplier saturates near the blast centre and diffuses out at the edges within a few substeps,
  so the cloud lacks visible texture. Needs more dramatic initial structure. Open.
- **Per-channel smoke attenuation** — the *light* march uses a single scalar `smoke_absorption`;
  the colour of god-rays comes from the deposited light's colour, not from three independent smoke
  coefficients. Coloured smoke (e.g. a tinted gas that absorbs selectively per channel) would need
  a per-channel `smoke_absorption`, which the current model does not provide. Not a defect — the
  shipped model is monochrome-absorbing — but a named extension point: this per-channel signature *is*
  the colour model of the multi-gas system (§6).
- **No smoke substep stability cap** — the design relies on wind-dependent diffusion to suppress
  advection oscillation rather than a hard CFL limit; large `advection_rate` or `dt_scale` values
  can still oscillate. Consistent with the design intent but worth noting for tuning.
