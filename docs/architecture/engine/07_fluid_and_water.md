# Fluid & Water

**Depends on:** [Grid & Coordinates](01_grid_and_coordinates.md), [State & Ownership](02_state_and_ownership.md) (GameMap)

---

## 1. What this system is

The fluid system is the ship's **liquid layer**: a depth of water (and, later, oil)
that pools on the floor, runs downhill, sloshes when the ship rocks, and floods
compartments through breaches and burst pipes. It is the third bulk physical medium
in Breach, sitting alongside the atmosphere and the temperature/fire field. Where
the atmosphere fills the room with gas, the fluid layer covers the floor with liquid.

The design target is the set of scenes that make a damaged ship feel alive:

- An aquarium's glass shatters and the tank empties across the deck, the water
  finding its way around furniture and through doorways.
- A pipe bursts and feeds a steady stream that slowly floods a compartment.
- The hull is holed below the waterline and the sea pours in.
- The ship lists from battle damage and the standing water all slides to the low
  side — the *Titanic* effect — draining one room and flooding the next as the tilt
  progresses.

None of these are scripted set-pieces. Like the atmosphere, they fall out of a
single solver reading and writing one shared depth field under one set of boundary
rules. Where the water goes is decided by gravity, by the floor's shape, and by the
ship's tilt — not by a designer placing it.

**Status note.** This system is designed and prototyped but **not integrated**. The
solver exists and is visually validated in `prototypes/archive/` (see §8); it is not
yet a C++ solver, has no fields on `GameMap`, and is not in the physics tick. This
chapter is the canonical design the integration should follow, written so the
prototype's choices land as shipped contracts. Everything in §2–§6 is the intended
design; §8 audits honestly what exists today.

---

## 2. The model — pipe + damped velocity

### 2.1 The state

The fluid is one primary field over the tile grid:

| Field | Meaning | Character |
|-------|---------|-----------|
| `water_depth` | depth of liquid standing on the floor, in metres | slow: flows, pools, conserves mass |

plus a velocity auxiliary (`flow_vx`, `flow_vy`) that carries momentum between
ticks, exactly as the atmosphere's `wave_v` does for the wave field. Depth is the
quantity gameplay and rendering read; velocity exists only to make the flow look and
behave like a fluid rather than instantly levelling.

The terrain the fluid flows over is the **floor height map** — a static per-tile
height that is the single source of truth for floor geometry (it also generates the
lighting normal map; see the Graphics chapter). Drain grooves, raised thresholds,
furniture legs, and sloped corridors are all just features of this height field, so
the water reacts to them automatically without any of them being modelled as a
special case.

### 2.2 The update

The solver advances `water_depth` and the velocity field one step with three lines
of physics:

```
surface          = height_map + tilt_offset + water_depth
flow_velocity   += dt · (−g · ∇surface − damping · flow_velocity)
water_depth     -= dt · div(flow_velocity · water_depth)      # upwind flux
```

- **Surface** is the height of the top of the water: the floor, plus the tilt offset
  (§2.3), plus the standing depth. This is the potential the fluid falls down.
- **Velocity** accelerates down the surface gradient (`−g·∇surface`) and is bled off
  by linear damping. Damping is what makes the model a *pipe* model rather than an
  inviscid one: it is the friction that lets water settle into a flat pool instead of
  oscillating forever.
- **Depth** changes by the divergence of the mass flux `velocity · depth`, computed
  with **upwind** face selection — the depth carried across a cell face is taken from
  the cell the flow is coming *from*. Upwinding is what keeps the scheme from
  oscillating at sharp wet/dry fronts.

The gradient uses central differences in the interior with Neumann (mirrored)
boundaries at walls, the same stencil convention as the atmosphere Laplacian. After
each step, velocity is zeroed on walls, fluxes are zeroed across any face touching a
wall, and depth is clamped non-negative and zeroed on walls. Those three masks are
the entire wall interaction: water cannot accelerate into a wall, cannot flux through
one, and cannot stand inside one.

### 2.3 Ship tilt

Tilt is a per-tile height offset added to the terrain, computed from the ship's two
tilt angles about its centre:

```
tilt_offset(x, y) = tan(tilt_x) · (x − cx) · dx
                  + tan(tilt_y) · (y − cy) · dx
```

Because the offset is added to the floor height *inside* the surface gradient, tilt
is just terrain that changes over time. A sinusoidal `tilt_x` is a ship rocking on
swell; a slowly growing tilt from accumulating hull damage is the *Titanic* list,
and water rushes to the low side through whatever corridors and doorways connect to
it. No special "flow toward the low side" code exists — the gradient already points
there.

### 2.4 Fluid sources

Sources are not part of the solver step; they are writes into `water_depth` between
steps, mirroring how explosions are events that stage energy for the atmosphere
solver rather than living inside it:

- **Aquarium / tank burst** — a body of standing water released when its containing
  wall is destroyed. This is just `destroy_wall` on the glass; the released column
  then flows under its own gradient.
- **Burst pipe** — a continuous source: each tick, hold one or more tiles at a
  target depth (`depth = max(depth, source_level)`).
- **Hull breach below the waterline** — a continuous, pressure-dependent inflow at
  the holed tiles, the water analogue of the air *draining out* through the same
  breach.

The aquarium-burst, maze-flood, continuous-pipe, and tilted-ship source patterns are
all demonstrated in the prototypes.

---

## 3. Resolution

The pipe model is cheap — on the order of ten arithmetic operations per tile per
step, unconditionally stable, run one or two steps per frame. That cheapness is what
makes the resolution decision: the sim runs at the **fine-tile grid** the rest of the
world state lives on (`GameMap` is sized at fine resolution from the loaded level),
sharing the same height map the renderer already uses for normals. There is no
separate coarse physics grid for fluid. Water reacts to every floor feature the art
encodes, because the art's height map *is* the terrain.

A documented fallback exists if per-tile simulation proves too expensive on weak
hardware: run one depth value per **coarse** physics tile, and at render time compare
each fine pixel's height-map value against its tile's water level — pixels below
appear submerged, pixels above dry. This buys pixel-accurate visual flooding at
coarse-grid simulation cost, trading away only sub-tile flow channelling. It is a
tuning option, not the default.

The simulation is **sparse**: only tiles that currently hold (or border) fluid are
updated. A ship that has taken no water costs nothing.

---

## 4. Why this model, and not shallow water

The obvious "correct" choice for 2D liquid is the **shallow-water equations** — depth
plus conserved momentum, with a proper flux solver. It was implemented and tested
side by side with the pipe model (both live in the prototypes). It was **rejected**.

The shallow-water equations need an explicit flux scheme (Lax–Friedrichs was used)
with a CFL substep limited by the surface-wave speed `√(g·h)`. At the wet/dry
boundaries that *dominate* a flooding ship — every advancing waterfront, every tile
where water piles against a wall — that scheme diverges. The momentum conservation it
buys is not worth the instability for a game whose fluid needs to look convincing,
not be quantitatively accurate.

The pipe model gives up exact momentum conservation and gets, in return:

1. **Unconditional stability.** The damping term removes the CFL constraint entirely.
   There is no wet/dry blow-up. `dt` is chosen for visual smoothness (10–16 ms,
   one or two steps per 60 fps frame), not forced small by stability.
2. **Mass conservation.** Depth changes only by flux divergence, so total water is
   conserved to round-off — water neither vanishes nor is created.
3. **Cost.** Roughly ten operations per tile, cheap enough to run at fine resolution.

It produces visually convincing pooling, flow through corridors, sloshing under tilt,
and progressive flooding — which is the entire requirement. This mirrors the
atmosphere chapter's stability story: there, implicit diffusion removes the diffusion
CFL; here, velocity damping removes the surface-wave CFL. Both buy unconditional
stability by accepting an approximation that is invisible at game scale.

---

## 5. Coupling into the rest of the engine

The fluid layer is designed to plug into the existing physics tick the same way smoke
and fire do: a single-step solver, orchestrated from Python, sharing fields with its
neighbours rather than reimplementing them. Three couplings are designed.

### 5.1 Water → atmosphere: volume displacement

This is the most important coupling, and it needs almost no new machinery, because the
water surface already tells the atmosphere everything it needs.

Air is ~1000× less dense than water, so the coupling is **one-directional**: water
ignores air (gravity and the floor drive it), and air reacts to water (water decides
how much volume is left for air). The water step runs first; the atmosphere step then
reads the new depths. No two-phase solver is needed.

Each cell has a floor-to-ceiling capacity. Water takes some of it; air gets the rest:

```
air_volume[i] = cell_capacity[i] − water_depth[i] · cell_area
P[i]          = air_mass[i] · R · T / air_volume[i]      # ideal gas law
```

Rising water shrinks the available volume, which raises the pressure, which drives
airflow through the openings the atmosphere solver already handles — and smoke and gas
ride that flow as the passive scalars they already are. The order in the tick is:

| Step | What happens |
|------|--------------|
| 1 | **Water pipe model** runs — updates `water_depth` per cell |
| 2 | `air_volume = capacity − water_depth · area` per cell |
| 3 | air pressure from `P = nRT/V` |
| 4 | **Atmosphere solver** runs, with per-cell pressure influenced by reduced volume |
| 5 | Smoke / gas advection rides the resulting wind, unchanged |

Steps 2–3 are the only new work; everything else already exists. Two consequences fall
out cleanly. A cell whose water reaches the ceiling (`water_depth ≥ capacity` →
`air_volume = 0`) becomes a **wall for the atmosphere** — a fully flooded corridor
correctly blocks airflow. And at a hull breach below the waterline, water comes *in*
while air goes *out*: the rising depth compresses the remaining air, briefly spiking
its pressure before it vents through the same hole — dramatic and physically correct,
with no special-case code.

The marquee emergent scene: the ship lists, water slides to starboard, starboard air
volume shrinks, pressure spikes, and smoke that the crew thought was contained gets
shoved through doorways to port. Two systems that each already exist, producing a
result neither was written to produce.

*(Open question carried from the design: whether the displacement pressure should
enter the atmosphere as a source term in its wave equation or run as a separate
equalisation pass. The source-term form is the leaning choice — cleaner, one pass.)*

### 5.2 Water + electricity: conduction

Water spreads electrical hazards. The intended hook: when an electrical arc (a
lightning bolt, a damaged power conduit) strikes water, flood-fill the connected body
of wet tiles and apply damage to every unit standing in it. Arc target priority is
metal > water > unit — current takes the most conductive path available — so a unit
in a puddle near exposed wiring is in real danger, and a unit on dry deck a tile away
is not. This makes standing water a tactical liability, not just a movement nuisance.

### 5.3 Fluid types and fire

Water ships first. **Oil** is the planned second type, and its point is flammability:
oil on the floor plus fire equals a spreading inferno across the deck, the fluid layer
feeding the fire layer. The fluid type is stored as a tag per tile. Stacking (oil
floating on water, sealing the surface and trapping air below) is a real interaction
the model permits but is deferred — water-only first, then oil, then layering.

### 5.4 Gameplay effects

Beyond the physical couplings, the depth field is read by gameplay:

- **Movement penalty** — deeper water slows units; past a threshold a corridor is
  impassable.
- **Electrical hazard** — §5.2.
- **Oil ignition** — §5.3.

---

## 6. Surface rendering

The simulation produces a depth field; making it *look* like water is a render-side
concern. The planned approach augments the settled water surface with slow ripples —
a low-speed wave overlaid on the depth, either a few summed sine waves or a cheap
surface-wave equation. The physical surface-wave speed `√(g·depth)` is ~1–3 m/s, two
orders of magnitude slower than the atmosphere's acoustic waves, so this is nearly
free. This is purely visual; it does not feed back into the flow.

A separate, unrelated render effect worth distinguishing here: **blood splats** and
similar stains are *not* the fluid sim. They are decals on the destruction paint layer
(see the Graphics chapter), painted by unit-damage events. They look wet but carry no
depth and never flow. The fluid system is only the simulated liquid layer; stains are
paint.

**Forward — water-optics research pass (Erik's request, 2026-06-07).** When we build the
water surface for real, do a realism-research pass on its optics and rendering the same
way we did for smoke and gases (see Smoke ch.05 §6.1): Fresnel reflection, refraction,
depth-tint, foam at wet/dry fronts, caustics, and normal / ripple maps for the surface.
The notes above are the placeholder; the dedicated pass is what should land the shipped
look.

---

## 7. Forward path

The integration, in the order the design implies:

1. **Port the pipe model to C++** as a single-step solver with the same shape as
   `AtmosphereSolver` / the smoke solver: construct with grid + parameters, expose one
   `step(dt)`, let Python own the substep loop.
2. **Add fields to `GameMap`** — `water_depth` and the `flow_vx`/`flow_vy` auxiliary,
   plus the floor `height_map` as the shared terrain — following the array-plus-mask
   state contract (no tile objects; accessed as `gmap.<field>`).
3. **Wire sources** to existing topology events: `destroy_wall` releases tanks,
   designated tiles act as pipe/breach sources.
4. **Insert into the physics tick** before the atmosphere step, so volume displacement
   (§5.1) reads fresh depths.
5. **Add the couplings** — volume displacement first (highest payoff, least code),
   then conduction, then oil + fire.
6. **CUDA residency** — the stencil and flux passes are local 2D operations and port to
   GPU unchanged, alongside the atmosphere and smoke fields when state goes
   GPU-resident.

---

## 8. Implementation status

Audited against the codebase: `prototypes/archive/fluid_test.py`,
`fluid_sandbox.py`, `fluid_scenarios.py`, `fluid_tilted_ship.py`;
`src/simulation/gamemap.py`; `src/simulation/environment.py`;
`src/simulation/physics_runner.py`.

**Status: prototype-only. Not integrated into the game.**

**Built (prototype, Python/NumPy, `prototypes/archive/`):**

- The pipe + damped-velocity solver itself — surface gradient, damped velocity update,
  upwind mass-flux divergence, wall masking, non-negative depth clamp. This is the
  model §2 describes and is consistent across all four prototype files.
- The rejected shallow-water solver (Lax–Friedrichs flux, CFL substepping), kept
  alongside the pipe model for the side-by-side comparison that justified the choice
  (`fluid_test.py`, `fluid_sandbox.py`).
- Ship tilt as a time-varying terrain offset, including the progressive-tilt *Titanic*
  flooding through connected rooms (`fluid_tilted_ship.py`).
- Source patterns: aquarium/dam burst on wall removal, continuous pipe source, maze
  flood-through, hull-breach inflow (`fluid_scenarios.py`, `fluid_tilted_ship.py`).
- Visual validation only — the prototypes render to Matplotlib animations / GIFs
  (`prototypes/fluid_pipe_only*.gif`, `fluid_test.gif`). No headless solver API.

**Designed but not built:**

- **C++ single-step solver.** None exists. The shipped atmosphere and smoke solvers
  are C++ in `cpp/`; the fluid solver is still NumPy in a prototype.
- **`GameMap` fields.** There is **no** `water_depth`, no flow-velocity field, and no
  floor `height_map` on `GameMap`. (`GameMap` carries `atmosphere`, `smoke`,
  `light_rgb`, `light_dir`, `heat`, `smoke_glow`, and the material masks — no fluid.)
- **Physics-tick integration.** `PhysicsRunner.step` orchestrates atmosphere, smoke,
  and fire only. The fluid solver is not called.
- **Water → atmosphere volume displacement** (§5.1) — design only; no `cell_capacity`,
  no volume/pressure coupling exists.
- **Water + electricity conduction** (§5.2), **oil / fluid types** (§5.3), **movement
  penalty and other gameplay reads** (§5.4), **surface-wave rendering** (§6) — all
  design only.
- **Coarse-grid + per-pixel-visual fallback** (§3) — documented option, not built.
- **CUDA residency** — forward idea.

**Gaps / things to resolve at integration time:**

- **Terrain source.** The pipe model needs a per-tile floor height map. The art
  pipeline designates the floor layer's height map as that terrain (and as the lighting
  normal-map source), but no height-map field is loaded onto `GameMap` today. This is
  the prerequisite the integration depends on, and it is unbuilt.
- **Naming collision to avoid.** `environment.py` already defines `REQUIRES_WATER` and
  `can_breathe_water` — these are **creature trait** flags (a fish needs water to
  breathe), entirely unrelated to the fluid sim. Fluid fields should not reuse this
  vocabulary.
- **Tick ordering is load-bearing.** Volume displacement requires the water step to run
  *before* the atmosphere step within the tick; this ordering must be honoured when the
  solver is inserted into `PhysicsRunner`.
- **Prototype `dx` values vary.** The prototypes use `dx` of 1.0, 0.33, and 1/3
  depending on scenario; the real solver must take the world's fine-tile size (1/3 m)
  rather than a hardcoded constant.
- **Source-term vs. equalisation-pass** for displacement pressure (§5.1) is an open
  modelling choice to settle when the coupling is implemented.
