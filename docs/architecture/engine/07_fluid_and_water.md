# Fluid & Water

**Depends on:** [Grid & Coordinates](01_grid_and_coordinates.md), [State & Ownership](02_state_and_ownership.md) (GameMap); the phase transitions (§5.4) additionally depend on [Temperature & Fire](06_temperature_and_fire.md)

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

**Status note.** The core of this system is **built and integrated** (2026-06-10,
commits `3ebf62b`…`cb229c7`): the C++ pipe-model solver, the `GameMap` fields, the
physics-tick insertion, the volume-displacement and pressure-head couplings, the
flash-boil sink with its steam puff, the ripple field, and the debug-render layer all
ship green (277 tests). Built from this chapter via the two-round-reviewed
`docs/water_implementation_plan.md`. §8 audits exactly what shipped and what remains
(conduction, oil, ice, gameplay reads, the optics research pass).

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

The terrain the fluid flows over is the **floor height field** (`floor_height`) — a
static per-tile height. Drain grooves, raised thresholds, furniture legs, and sloped
corridors are all just features of this height field, so the water reacts to them
automatically without any of them being modelled as a special case. The field is
**optional**: it defaults to flat zero, and on a flat floor the model still delivers
pooling, tilt-sloshing, and doorway flow — water ships before any art height map
exists. When the art pipeline does deliver one (the same data that generates the
lighting normal map; see the Graphics chapter), it simply becomes the terrain (§3).

### 2.2 The update

The solver advances `water_depth` and the velocity field one step with three lines
of physics:

```
surface          = floor_height + tilt_offset + water_depth
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

**Units never block water** (decided 2026-06-10). The wall mask is the *static* solid
mask only: a unit in a doorway is washed over, not a dam — it would look absurd for a
standing body to hold back a flooded room. Partial unit drag was considered and
rejected: the natural vehicle (`dyn_permeability`, where units sit at 0.5 for air)
is also written to 0 on fully-flooded cells by the displacement coupling (§5.1), so
reusing it for water flux would make flooded cells block their own water. Units
interact with water the other way around: depth slows them, and strong flow pushes
them (§5.5).

One coupling later adds a fourth term to this same potential: air pressure as a
**head term** (§5.1), so blasts and decompression shove water through the identical
gradient machinery — no second flow mechanism exists.

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
- **Hull breach below the waterline** — at sea: a continuous, pressure-dependent inflow
  at the holed tiles, the water analogue of the air *draining out* through the same
  breach. In space the same hole runs the other way: exposed water flash-boils away
  into vacuum — a depth *sink*, not a source (§5.4).

The aquarium-burst, maze-flood, continuous-pipe, and tilted-ship source patterns are
all demonstrated in the prototypes.

---

## 3. Resolution

There is exactly **one grid** (ch.01): `water_depth`, like every other field, lives on it
at the level's `tile_size_m`, and the solver derives its constants from that `dx` — never
from a hardcoded 1/3 m. The pipe model is cheap — on the order of ten arithmetic
operations per tile per step, unconditionally stable, run one or two steps per frame — so
it simply runs where everything else runs. There is no separate fluid grid in either
direction; resolution is the *level's* fidelity/performance knob, not this system's.

The simulation is **dense**, like the atmosphere and smoke solvers, with one scalar
early-out: the runner tracks total water and skips the solver entirely while the ship is
dry. A ship that has taken no water costs nothing — no sparse active-set bookkeeping is
needed to get that, and none is used. (If profiling ever demands more, a wet-region
bounding box is the documented optimization lever; it is an implementation detail, not a
design commitment.)

**Terrain comes in two resolutions, and only one is simulated.** The solver reads the
per-tile `floor_height` field (§2.1), optional and defaulting to flat zero — water ships
before any height map exists. Separately, *if* the art pipeline delivers a height map at
art resolution, the **renderer** refines the picture per pixel: compare each pixel's
height against its tile's water surface — below appears submerged, above dry. That buys
pixel-accurate shorelines and puddle edges at zero simulation cost, and it is purely
visual: the depth that gameplay and physics read is the per-tile value. Sub-tile flow
channelling (water following a groove narrower than a tile) is the one thing this cannot
do, and we give it up knowingly.

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

### 5.1 Water ↔ atmosphere: volume displacement and pressure head

This is the most important coupling, and it needs almost no new machinery, because the
water surface already tells the atmosphere everything it needs.

Air is ~1000× less dense than water, so the coupling is **asymmetric**: water decides how
much volume is left for air, and air pushes back only through a deliberately small head
term (below). The water step runs first; the atmosphere step then reads the new depths.
No two-phase solver is needed — and **no air-mass field** either: both directions stay
entirely in the pressure formulation the atmosphere already uses.

**Water → air: volume displacement.** Each cell has a floor-to-ceiling air column; water
takes some of it, and air that keeps its mass in a smaller volume rises in pressure.
Isothermal compression, applied multiplicatively between the two solver steps:

```
free_h[i]      = ceiling_h − water_depth[i]              # the cell's remaining air column
atmosphere[i] *= free_h_before[i] / free_h_after[i]      # isothermal P·V = const  (ratio capped)
```

(`ceiling_h` is a per-level constant for now, per-tile later if a level ever wants it.)
Rising water shrinks the column and the pressure climbs; the gradient drives airflow
through the openings the atmosphere solver already handles — and smoke and gas ride that
flow as the passive scalars they already are. The order in the tick:

| Step | What happens |
|------|--------------|
| 1 | **Water pipe model** runs — updates `water_depth` per cell |
| 2 | **Volume scaling** — `atmosphere[i] *= free_h_before / free_h_after`, ratio capped |
| 3 | **Atmosphere solver** runs — implicit diffusion equalises the displacement bump; wind follows |
| 4 | Smoke / gas advection rides the resulting wind, unchanged |

**Resolved (2026-06-09): there is no separate equalisation pass.** The atmosphere's
implicit diffusion *is* the equaliser, and pressure equalising across cells of different
free volume is the physically correct equilibrium. Step 2 is the only new work.

Consequences fall out cleanly. A cell flooded to the ceiling (`free_h ≤ 0`) becomes a
**wall for the atmosphere** — a fully flooded corridor correctly blocks airflow. At a
hull breach below the waterline, water comes *in* while air goes *out*: the rising depth
compresses the remaining air, briefly spiking its pressure before it vents through the
same hole — dramatic and physically correct, with no special-case code. And the rule is
symmetric: *receding* water leaves its cell under-pressured, so air rushes back into a
draining compartment — including into a re-opened flooded cell, which re-enters the
atmosphere near zero and fills by the same diffusion.

The marquee emergent scene: the ship lists, water slides to starboard, starboard air
volume shrinks, pressure spikes, and smoke that the crew thought was contained gets
shoved through doorways to port. Two systems that each already exist, producing a
result neither was written to produce.

**Air → water: the pressure head.** The reverse coupling is one term added to the
potential the water already falls down (§2.2):

```
surface = floor_height + tilt_offset + water_depth + k_p · (atmosphere + wave_p)
```

Constant pressure vanishes under the gradient, so a uniform 1 atm changes nothing — only
pressure *differences* push water, which is exactly right. Physically the head constant is
`k_p = 1/(ρ_w·g) ≈ 10.3 m` of water per atmosphere; the shipped `k_p` is tuned far lower,
because our `wave_p` rings for ~a second where a real blast's overpressure lasts
milliseconds — realistic head applied over game timescales would over-displace. What it
buys: a grenade over a flooded deck punches a **crater into the water and sends a ring
wave** outward (the §6 ripple field rings with it); decompression through a breach drags
the standing water toward the hole along with the air. Shockwaves in water — no new
solver, no air mass, one term in a potential we already compute. **Tuning/realism research
pass at implementation (Erik's request, 2026-06-09).**

**The whoosh (flourish, deferred).** Violent displacement can also deposit into
`wave_source` in proportion to the rate of depth change above a threshold, so fast
flooding is heard as pressure, not just seen as geometry. Cheap, optional, lands after the
core coupling ships.

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

### 5.4 Phase transitions: ice ↔ water → vapour

Water is the liquid middle of a phase triple, and both neighbours earn their place:
vacuum boils it, cold freezes it. The committed transitions are **ice ↔ water →
vapour**; closing the cycle (vapour back down) is a research-flagged stretch. All of
them are cheap local conversion rules over fields that already exist or are already
being built — none of them touches the solver.

- **Water → vapour** *(ships with the water core)*. Water exposed to near-vacuum
  pressure flash-boils: a rate-limited depth sink on exposed tiles — the §2.4
  space-breach behaviour. Once the gas layer exists, the boiled-off mass can enter it
  as a steam puff (visible venting); until then it simply leaves the world. When the
  temperature field lands, the same sink also runs at high temperature: fire boils
  shallow puddles dry.
- **Ice ↔ water** *(needs the temperature field, ch.06 — in flight)*. Where tile
  temperature crosses freezing, `water_depth` converts (rate-limited) into `ice_depth`
  — and **ice is terrain**: it stops flowing and adds to the effective floor height,
  so later water flows *over* it, and melting converts it back. No new solid state, no
  solver change — a transfer between a flowing field and the height field. Frozen
  tiles drop out of the §5.2 conduction body (ice doesn't conduct), and venting a
  flooded corridor to space can leave a walkable ice floor behind.
- **Vapour → water / ice — rain and snow** *(research)*. The sim rule is nearly
  trivial once steam is a gas species with a temperature: cold steam deposits back as
  `water_depth` (rain) or, below freezing, as `ice_depth` (snow). The open half is
  **rendering** — droplet particles, rainfall as ripple-field sources, white albedo
  for snow accretion. Indoor weather on a dying ship is worth the research pass; it is
  not load-bearing for anything above. **(Erik's request, 2026-06-09.)**

### 5.5 Gameplay effects

Beyond the physical couplings, the depth field is read by gameplay:

- **Movement penalty** — deeper water slows units; past a threshold a corridor is
  impassable.
- **Washed along** — strong flow pushes units along `flow_v` (a unit in a draining
  doorway goes *with* the water). The same read-the-gradient pattern as the designed
  decompression suction (ch.04 §5) — both unbuilt; they should land together.
- **Electrical hazard** — §5.2.
- **Oil ignition** — §5.3.
- **Ice underfoot** — a frozen tile reads as solid floor (flooded-impassable becomes
  walkable) and optionally as slippery; a flourish that lands with §5.4's second stage.

---

## 6. Surface rendering

The simulation produces a depth field; making it *look* like water is a render-side
concern, built from **two layers with distinct jobs** — and one hard rule shared by
both: **neither feeds back into the flow.** The transport gradient (§2.2) never sees
them. Feeding surface waves back into depth would reinvent the shallow-water wet/dry
instability through the back door — the exact failure the pipe model was chosen to
avoid (§4).

- **The ripple field — a real, but visual-only, wave.** A `ripple` displacement field
  with its velocity auxiliary, advanced by the same damped kick-drift wave update the
  atmosphere uses (`v += dt·(c²·Δripple − γ·v)`; `ripple += dt·v`), with
  `c² = g·min(depth, h_cap)` and both fields zeroed on dry tiles and walls. Its sources
  are events: a `wave_p` blast passing over wet tiles (§5.1), breach inflow, units
  wading — so explosions splash, waves ring off walls, and wakes trail through flooded
  corridors, all emergent from one stencil.
  - **The cap is physics, not a fudge.** `√(g·depth)` is the *shallow-water* speed,
    valid while depth ≪ wavelength; once `depth ≳ λ/2π`, real waves go depth-blind
    (deep-water dispersion). The grid resolves ripples of λ ≈ 3–6 tiles (~1–2 m), so
    `h_cap = λ/2π ≈ 0.2–0.3 m` splices the two regimes — and pins `c_cap ≈ 4–5 tiles/s`
    against the atmosphere's 66. The solver derives a **static** `max_dt = 0.5/c_cap`
    at init, exactly as the atmosphere does: no adaptive stepping, **one substep per
    tick at any tick rate we use**.
  - **Shoaling and refraction are free** — `c` varies with depth, so ripples slow and
    bunch up entering shallows and wavefronts bend toward them. True *breaking* is
    deliberately out: it is the nonlinear regime (`c` read from `depth + ripple`, crest
    overtaking trough, shock formation), the classic explicit-scheme divergence — and
    this field is linear precisely so it cannot steepen into a shock. The look is
    bought cheaply instead: clamp `|ripple| ≤ k·depth` (waves no taller than the water
    — also a hard amplitude guarantee), and render **foam where `|∇ripple|` exceeds a
    threshold** — whitecaps at the steep fronts (feeding the optics pass below). The
    wet/dry edge is self-absorbing (`c² → 0` plus damping), so shores dissipate
    arriving ripples the way beaches do.
- **Ambient ripples — shader-side sine waves.** Standing water needs idle texture; a
  few summed sines in the shader provide it. The modulation rule is deliberately
  simple: ambient amplitude = a base level + the local ripple-field energy. The two
  layers then read as one surface that gets agitated when something happens nearby and
  calms down afterwards.

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
   `AtmosphereSolver` / the smoke solver: construct with grid + parameters
   (`dx = tile_size_m` from the level), expose one `step(dt)`, let Python own the
   substep loop.
2. **Add fields to `GameMap`** — `water_depth` and the `flow_vx`/`flow_vy` auxiliary,
   following the array-plus-mask state contract (no tile objects; accessed as
   `gmap.<field>`). `floor_height` is optional from day one: flat zero until the art
   pipeline delivers it.
3. **Wire sources** to existing topology events: `destroy_wall` releases tanks,
   designated tiles act as pipe/breach sources, vacuum exposure runs the flash-boil
   sink (§5.4).
4. **Insert into the physics tick** before the atmosphere step, so volume displacement
   (§5.1) reads fresh depths.
5. **Add the couplings, in payoff order** — volume displacement with its capped scaling
   first (highest payoff, least code), then the pressure-head term (blasts shove
   water), then conduction, then oil + fire.
6. **Surface presentation** — the ripple field and ambient sines (§6), and the
   per-pixel render refinement once an art-resolution height map exists (§3).
7. **Phase transitions, staged** (§5.4) — flash-boil ships with the core; ice ↔ water
   once the temperature field lands; rain/snow after the research pass.
8. **CUDA residency** — the stencil and flux passes are local 2D operations and port to
   GPU unchanged, alongside the atmosphere and smoke fields when state goes
   GPU-resident.

---

## 8. Implementation status

Audited against the shipped code (2026-06-10): `cpp/src/water_solver.{h,cpp}`,
`cpp/src/bindings.cpp`, `src/simulation/gamemap.py`, `src/simulation/physics_runner.py`,
`src/simulation/field_edit.py`, `config.toml [physics.water]`, `renderer/overlays.py`,
`tests/test_water_*.py`. Build record: `docs/water_implementation_plan.md` (two-round
review + per-step commits, suite 228 → 277 green).

**Status: core built and integrated.** Water flows, pools, slides under tilt, displaces
air, gets shoved by blasts, boils in vacuum, and ripples — in the game tick, dormant-safe
(a dry ship pays one `.any()`).

**Built and shipped (W1–W6b, commits `3ebf62b` → `cb229c7`):**

- **C++ `WaterSolver`** (§2) — surface-gradient + damped-velocity pipe model, donor-cell
  upwind flux with a per-cell outflow limiter (mass-exact), Neumann mirror at the static
  `solid` mask (units never block water), `dx` from the level's `tile_size_m`, real wave
  CFL via `max_dt()` (substeps derived house-style; 2/tick at 24 tps, 3 with the head on).
- **`GameMap` fields** — `water_depth`, `flow_vx/vy`, `floor_height` (optional, flat-zero
  default), `ripple`, `ripple_v`, `tilt_x/tilt_y`, `tile_size_m`, `water_sources`;
  `water_depth` registered in the FieldEdit policy table (ADD/REMOVE, clamped ≥ 0).
- **Tick insertion** (§5.1 ordering) — `PhysicsRunner._step_water` right before the IMEX
  loop: source holds → substeps → flash-boil → displacement accounting (persistent
  end-of-previous-tick snapshot, so FieldEdit dumps and holds are each counted once).
- **Volume displacement** (§5.1) — multiplicative isothermal free-column scaling, ratio
  capped, flooded cells seal airflow via `dyn_permeability = 0` (face-flux blocking;
  trapped `wave_p` decays rather than being zeroed — by design).
- **Pressure head** (§5.1) — live at `k_p = 0.5` (`config.toml`); blasts crater pools,
  venting drags water. Feel-tuning owed (Erik's research flag): worst case is
  near-flooded cells, `c_eff = √(g·h·(1 + k_p·P/free_h))`.
- **Flash-boil + steam** (§5.4 stage 1) — pressure-keyed (`atmosphere < boil_p_thresh`)
  depth sink; boiled mass enters `gas[white_smoke]` via `steam_yield` (the constant the
  fire-side evaporation branch must reuse — see `07_notes_from_claude.md` Answers).
- **Ripple field** (§6) — `step_ripple` damped kick-drift wave, `c² = g·min(depth, h_cap)`
  (deep-water splice), amplitude-clamped, one call per tick, splash-fed by `wave_p` over
  wet tiles; proven visual-only by a 60-tick A/B rollout (transport bit-identical).
- **Debug layer** — U pour / O depth overlay / P tilt nudge; overlay v2 with ripple
  shading, foam from `|∇ripple|` + wet/dry-front speed, ambient sines whose amplitude
  tracks local ripple energy (`[display] water_*` knobs, restart-bound).

**Prototype record (historical, `prototypes/archive/`):** the NumPy pipe model + the
rejected shallow-water side-by-side (`fluid_test.py`, `fluid_sandbox.py`), tilt flooding
(`fluid_tilted_ship.py`), source patterns (`fluid_scenarios.py`). One deliberate
divergence from them shipped: walls use the canon Neumann mirror, not the prototypes'
tall-terrain fake (§2.2; tested by the settled-pool-flat-at-wall property).

**Designed but not built:**

- **Water + electricity conduction** (§5.2), **oil / fluid types** (§5.3).
- **Phase transitions beyond flash-boil** (§5.4) — ice ↔ water (gated on temperature
  tuning; `ice_depth` does not exist yet), rain/snow (research pass owed).
- **Fire-side evaporative heat sink** — the other half of the §5.4/fire interface; lives
  in the temperature cooling pass (fire side's lane), unblocked now that `water_depth`
  exists.
- **Gameplay reads** (§5.5) — movement penalty, washed-along, ice underfoot.
- **Per-pixel render refinement** (§3) and the **water-optics research pass** (§6) —
  the shipped overlay is the placeholder look.
- **The whoosh** (§5.1 flourish), **CUDA residency**, engine-wide fixed-point.

**Standing facts (kept from the design audit):**

- **Terrain is optional** — `floor_height` defaults flat zero; the art height map, when
  it lands, becomes the terrain (and enables the §3 per-pixel refinement).
- **Naming** — `REQUIRES_WATER` / `can_breathe_water` in `environment.py` are creature
  traits, unrelated to the fluid sim; the vocabularies stay disjoint (honoured).
- **Tuning dials for the feel session:** `[physics.water]` `damping` (prototype-validated
  range 0.5–1.0), `k_p` (0.5), `boil_rate`/`steam_yield`, ripple `gamma_r`/`k_amp`/
  `k_splash` (pure feel), `[display] water_*` (overlay look).
