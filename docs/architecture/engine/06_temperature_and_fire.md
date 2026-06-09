# Temperature & Fire

**Depends on:** [Grid & Coordinates](01_grid_and_coordinates.md), [State & Ownership](02_state_and_ownership.md), [Material System](03_material_system.md), [Atmosphere & Pressure](04_atmosphere_and_pressure.md), [Ray Engine](08_ray_engine.md).

This chapter describes how Breach turns deposited heat into temperature, how
temperature ignites and destroys the world, and how fire lives, spreads, and
dies on the ship. It is the consumer end of the ray engine: rays carry energy
across air, this system decides what that energy *does* to solids and units.

The system has three layers, listed from the most physical to the most
gameplay-scripted:

1. **Heat → temperature.** Radiation lands in the `heat` buffer; a conduction
   pass turns accumulated heat into a `temperature` field that lives on solids.
2. **Temperature → world reactions.** Temperature crossing material thresholds
   ignites flammable tiles, melts walls, burns smoke away, and damages units.
3. **Fire.** A self-contained cellular system on flammable tiles: it spreads,
   consumes oxygen, emits smoke, is steered and modulated by wind, and burns
   walls through.

Today layer 3 is fully built and running; layers 1 and 2 are designed and the
plumbing they need (the `heat` buffer, the `conductivity` cache, the
`ignition_temp` column) is shipped but not yet consumed. The
**Implementation status** section at the end is explicit about the seam.


## 1. Where heat comes from, where temperature lives

**Heat crosses air as radiation, never as an air-temperature field.** This is
the single foundational choice of the system. A fire, an energy beam, or an
explosion deposits energy by *casting rays* (see Ray Engine); the rays write a
per-tile `heat` buffer wherever their energy lands. There is no diffusing
"air gets warm" field — heat only becomes *temperature* on solids.

This is deliberate, and every gameplay thread depends on it:

- **Radiation ignites at a distance.** A beam radiates heat across an open room;
  distant wood crosses its ignition threshold and catches — no scripting, no
  line-of-fire special case, because the ray already deposited the heat there.
- **Wind drives firestorms through fire, not air temperature.** Wind intensifies
  fire (§5); a hotter fire radiates further (as a brighter ray source); that
  ignites more fuel. The cascade is real because fire is a ray emitter, not
  because warm air advects.
- **Air temperature would ignite nothing and advect everything** — the wrong
  model for a tile game about fire spreading along walls and across gaps.

**Temperature lives on solids only.** It is implemented as a dense, full-grid
field with **conductivity = 0 on air**, so air tiles are no-ops at ambient and
hold no meaningful temperature. The field is dense rather than sparse for two
reasons: a dense layout is GPU-friendly (no scatter/gather) and the full-grid
structure leaves the door open to a future temperature→pressure coupling
(thermal-expansion firestorms) as a cheap additive change rather than a
rewrite.

```
  fire / beam / explosion
          │  (cast rays — Ray Engine)
          ▼
   heat   buffer        per-tile deposited energy this tick  (Q16.16 int)
          │  (conduction pass — this chapter)
          ▼
 temperature field      lives on solids; κ=0 on air         (fixed-point int)
          │
          ├──► ignition        temperature ≥ ignition_temp ∧ O₂ → start fire
          ├──► wall failure    temperature ≥ material limit → deplete wall_hp
          ├──► smoke burn-off   high heat removes smoke (laser tunnels)
          └──► unit damage      units sample heat at their tiles → env. damage
```


## 2. The conduction scheme — faked, unconditionally stable

The conduction that turns `heat` into `temperature` is **not** a physically
accurate heat equation. The goal is "beams glow, heat spreads along metal in a
way that reads well and reaches the ignition threshold," not thermodynamic
fidelity. Freed from physical accuracy, the design picks the cheapest scheme
that preserves the *qualitative* behaviour:

- **One relaxation pass per tick** — temperature relaxes toward a
  conductivity-weighted blend of its neighbours. There is **no CFL substep
  loop**. (An earlier draft sized the pass at ~17 CFL substeps per tick to keep
  metal's high conductivity stable; the relaxation scheme makes that
  unnecessary — it is unconditionally stable by construction.)
- **Power-of-two relaxation rates** — each material's per-tick relaxation rate
  is chosen as a negative power of two, so the fixed-point update is a **shift
  plus an add**, with no division. This is what makes integer temperature
  essentially free (§3) and removes the harmonic-mean division an honest
  diffusion would need.
- **κ = 0 on air → air tiles relax to nothing.** Heat does not conduct through
  air; it only crosses air as radiation (§1). Solid-to-solid conduction lets a
  laser hit on the hull spread heat fast along connected metal until it reaches
  an interior wood wall.

The only invariant the scheme must preserve is: *heat spreads along
high-conductivity material and accumulates toward the ignition threshold.*
Conductivity comes from the material table's `conductivity` column, projected
onto the grid as the per-tile `conductivity` cache (already built by the
GameMap). Relative values are tuned for gameplay, not measured:

| Material | `conductivity` | Behaviour |
|----------|---------------:|-----------|
| Hull     | 50.0  | Metal — heat races along a whole hull section; it glows as a unit |
| Steel    | 45.0  | Metal — like hull |
| Glass    | 1.0   | Middling |
| Door     | 0.3   | Wood-like |
| Wood     | 0.15  | Slow, local heating — stays hot where it was hit, eventually ignites |
| Air      | 0.0   | No conduction (radiation only) |

Setting `conductivity = 0` for a material turns it into a perfect insulator —
a deliberate, supported design lever.


## 3. Determinism — fixed-point where a value crosses a threshold

> **Principle:** fixed-point integers where a value crosses a discrete
> threshold into simulation state; float where it stays continuous and
> perceptual (rendering).

Two channels in this system feed gameplay thresholds, and both are
**fixed-point integers**:

1. **`heat` deposit.** Many rays land in one cell; integer addition is
   order-independent, so the accumulated deposit is identical regardless of the
   order rays are processed. On the GPU this becomes an `atomicAdd`; the integer
   property is what makes that deterministic.
2. **`temperature` field.** This is the value compared against `ignition_temp`
   and the wall-failure limit. It must be deterministic, and — because Breach
   leaves the door open to **lockstep multiplayer** — it must be **bit-identical
   across machines**, since lockstep exchanges only inputs and each machine runs
   the identical simulation. Float results would desync. Deterministic replays
   inherit the same benefit. The relaxation scheme (§2) makes fixed-point
   temperature nearly free, so there is no cost to pay for this.

The two determinism claims rest on different mechanisms, and that distinction
matters:

- **`heat` is deterministic by atomics** — order-independent integer addition.
- **`temperature` is deterministic by fixed rounding** — its relaxation is an
  atomic-free gather stencil, so its cross-machine identity rests on the
  power-of-two update being exact integer shifts, not on atomics. (A
  cross-machine bit-exactness test should validate this before lockstep is
  committed; the clean fallback, if that math ever proves painful, is *float
  temperature + integer heat-deposit* — single-machine determinism without the
  cross-machine guarantee.)

**Render-only channels stay float.** `light_rgb`, `light_dir`, and `smoke_glow`
have no downstream sim threshold (stealth and line-of-sight are image-based and
read the light buffers, not a quantized value), and fixed-point would band the
near-dark gradients at ambient and fight the HDR sum. Float is correct there.

### Fixed-point format (shipped)

The `heat` buffer is **Q16.16 int32** — 16 integer bits, 16 fractional bits, so
one unit of heat energy is `HEAT_SCALE = 65536` raw counts. The ray march:

- **quantizes** each float deposit into the fixed-point domain with
  round-to-nearest (`heat_quantize`), and
- adds it with a **saturating add** (`heat_saturating_add`): clamp at
  `INT32_MAX`, **never wrap**. Saturation protects the ignition threshold under
  a firestorm where many emitters dump energy into a few cells — the value pins
  at maximum rather than overflowing back to cold.

`ignition_temp` and the wall-failure limits are material-table thresholds; they
should be quantized into the same fixed-point domain **once at load** with a
pinned rounding mode, and the runtime test is `temperature ≥ quantized_limit`.
This load-time conversion is the single most determinism-critical step in the
system — it is fixed, never recomputed per tick.


## 4. What temperature drives

These are the field reactions — they *mutate the world*, so they run in the
deterministic sim step *after* the read-only ray pass has filled the buffers,
not inside the ray kernel.

**Ignition.** A flammable tile ignites when `temperature ≥ ignition_temp`
**and** oxygen is present (an `atmosphere` threshold on the neighbouring air).
Ignition starts a fire (§5), which thereafter consumes O₂ and emits smoke on
its own. `ignition_temp` is a per-material table value:

| Material | `ignition_temp` | |
|----------|----------------:|---|
| Wood | 300.0 | catches readily |
| Door | 280.0 | catches slightly sooner than bare wood |
| Hull / Steel / Glass / Air | 0.0 | non-flammable (never ignites) |

**Wall thermal failure.** When a wall's temperature crosses its material limit,
`wall_hp` depletes, and the tile is destroyed (converted to air) when it
reaches zero. This is how an energy weapon *melts through* a wall: the beam
deposits heat, conduction carries it into the wall, the wall heats past its
limit, `wall_hp` drains, the tile breaches. Because this routes through the
atomic-free temperature stencil rather than a direct write into `wall_hp` from
the kernel, it is deterministic — there is no separate "damage channel" to
deposit into, and no non-deterministic atomic write into `wall_hp`. Damage is
*derived* from temperature, not accumulated independently. (Wall destruction
goes through the GameMap's existing `destroy_wall` / `on_tile_changed` seam, so
the conductivity and occlusion caches stay consistent the instant a tile
breaches.)

**Smoke burn-off.** A field reaction removes smoke where heat is high. This is
what lets a laser tunnel a clear line through a smoke-filled corridor: the beam
deposits heat as it marches, and the burn-off reaction clears the smoke along
that path — while keeping the ray pass itself read-only.
**erik comment** im not sure if we want this inherent in the smoke system, or perhaps we want a methdo that can remove / add smoke at will at any coorinates- a method that  could be used by the laser weapons or whatever, and grenades alike. let's think it through togetehr.

**Unit damage.** Units take environmental heat damage by **sampling the `heat`
buffer at their footprint tiles** after the pass, in serial CPU unit logic
(deterministic by construction). A unit whose experienced heat pushes the
ambient temperature outside its `EnvironmentProfile` tolerance band
(`temperature_min`/`temperature_max`) takes `environmental_damage_rate` per tick
— the same channel a unit suffers in vacuum or out-of-range pressure. The
temperature a unit feels comes from radiation (the `heat` buffer it stands in),
not from a per-tile air-temperature field; this is exactly why the unit spec
defines temperature tolerance against *radiated* heat rather than an ambient
field. Energy-weapon splash and standing in a fire both flow through this one
path. (The unit reads the buffer; the kernel never writes the unit.)


## 5. Fire

Fire is the built, running layer. Each flammable tile carries a `fire` intensity
in `[0, 1]` (0 = unlit, 1 = full blaze), stepped once per tick by the C++
`FireSimulation`. **Spread is no longer cellular** — it comes entirely from
radiation → heat → temperature → ignition (§4 `apply_temperature_ignition`, fed
by fire's own heat-ray source, §"Fire as a light source"). The fire step is now
purely the *life and death* of an already-lit tile: a signed-logistic
**feedback** that grows a fed fire and decays a starved one
(`fire_design_proposal` §2/§3/§5). A tick runs these stages in order:

1. **Signed-logistic intensity feedback.** Per burning *flammable* tile of
   intensity `I`:

   ```
   T     = temperature[i] / TEMP_SCALE            # conduction-pass field (game units; wood ignites at 300)
   F     = clamp01(wall_hp[i] / fuel_ref)         # fuel from remaining wall HP
   P     = mean atmosphere over open (non-solid, non-vacuum) 4-neighbours   # O2 proxy
   W     = sqrt(wind_x² + wind_y²)                # the SHARED wind field (= −grad p, incl. shockwaves)
   hot   = clamp01((T − fire_T_ext) / fire_T_span)
   o2    = smoothstep(P_min, P_full, P)           # pressure IS oxygen — there is no O2 field
   avail = F · o2
   grow  = k_grow · avail · hot · I·(1−I) · (1 + k_wind_fan·W)     # logistic, wind-fanned
   die   = k_die · (1 − avail·hot) · I  +  k_wind_strip · W · (1−I) · I   # decay + wind blow-out
   I    += dt·(grow − die);  clamp01;  if I < I_min → 0            # snap-extinguish
   ```

   `I·(1−I)` gives accelerating growth at low `I` and saturation near 1. `hot`
   is the critical-flame-temperature brake (below `fire_T_ext` the fire dies even
   with fuel + O2). `o2` is the pressure-as-oxygen brake (a vented/low-pressure
   room reads `P → 0` → the fire suffocates — the **decompression-extinguishes**
   loop, now driven by a read, not a kill-threshold). `avail = F·o2` couples the
   fuel and O2 brakes. **Wind** (`W`) both *fans* growth (`1 + k_wind_fan·W` — a
   grenade shockwave is a transient wind spike that flares a blaze into a
   firestorm) and *blows out* small/marginal fires (`k_wind_strip·W·(1−I)·I` — the
   realistic crossover: the same gust that fans a big fire snuffs a guttering one).

2. **Own-tile plume pressure deposit.** Each burning tile adds a small
   self-limiting overpressure to its **own** tile (an order-independent own-cell
   write, *not* the deleted backwards subtraction that sucked smoke in):

   ```
   atmosphere[i] += max(fire_pressure_gain · I · (1 − atmosphere[i]/p_expand_ref) · dt, 0)
   ```

   So `wind = −grad p` points **outward** and the plume/smoke is pushed *away*
   from the flame. The sustain read `P` (stage 1) is the *neighbour* mean, so the
   fire reads incoming fresh air, not its own bump. A sealed-room fire
   over-pressurises → the existing `find_burst_walls` may pop a weak wall → the
   room vents → `P` falls → the fire starves. Emergent, no new code.

3. **Smoke emission.** Fire adds `smoke` to adjacent air tiles, which the smoke
   dynamics then advect on the wind.

4. **Wall burn-through.** Fire depletes `wall_hp` on the tile it burns; when a
   flammable wall reaches zero HP it is reported as destroyed. The fire step
   returns the list of `(y, x)` tiles that burned through; the per-tick runner
   calls `destroy_wall` on each (the solver itself never edits the material
   grid). **Burn-through *is* the fuel-consumption brake**: as `wall_hp → 0`,
   `F → 0` in stage 1, and the fire starves (burnout) before / as the wall fails.

Finally `fire` and `smoke` are clamped to `[0, 1]`; `atmosphere` is left
unclamped (the atmosphere solver owns its own bounds).

### Fire parameters

Set on the C++ `FireParams` struct, bound from `config.toml` `[physics.fire]`
(Erik tunes these live):

| Parameter | Default | Meaning |
|-----------|--------:|---------|
| `k_grow`             | 4.0   | Logistic growth gain (1/s) |
| `k_die`              | 2.0   | Decay rate when starved/cold (1/s) |
| `fire_T_ext`         | 350.0 | Extinction temperature (~`ignition_temp` + 50) |
| `fire_T_span`        | 150.0 | Width of the `hot` ramp above `fire_T_ext` |
| `fuel_ref`           | 60.0  | `wall_hp` normaliser: `F = clamp01(wall_hp/fuel_ref)` |
| `P_min`              | 0.60  | Pressure below which the O2 proxy is 0 |
| `P_full`             | 1.00  | Interior pressure where the O2 proxy is full |
| `I_min`              | 0.02  | Snap-to-zero extinguish floor |
| `k_wind_fan`         | 0.5   | `(1 + k_wind_fan·W)` fans growth — *needs tuning vs the wind scale* |
| `k_wind_strip`       | 0.5   | `W·(1−I)·I` blows out small fires — *needs tuning vs the wind scale* |
| `fire_pressure_gain` | 0.15  | Own-tile plume overpressure gain (1/s) |
| `p_expand_ref`       | 1.30  | Self-limiting plume saturation ceiling |
| `smoke_emission`     | 0.8   | Smoke produced per second per unit intensity |
| `wall_damage`        | 0.4   | Wall HP lost per second per unit intensity (the burnout brake) |

### Fire as a light source

Fire is *physical light*: a burning tile emits warm light through the same ray
engine as every other source — **no special fire-light code path.** The
discipline is to stay inside the physics sim and invent no exceptions. Cost is
controlled by per-source `max_range`: fire tiles cast **short** rays (roughly
`range = 2 + 2·fire_intensity`, `intensity = 0.3 + 0.7·fire_intensity`), so even
a flamethrower's worth of burning tiles is cheap — total cost is proportional to
total rays cast, and many sources × few rays each is the same budget as few
sources × many rays. Short-range raycasting (rather than painting a glow disc
around the flame) is what preserves correct wall occlusion: fire next to a wall
does not bleed light onto the wrong side.

This is also the bridge between fire and the heat system: because fire is a ray
source with a non-zero `heat` emission, every burning tile *already* deposits
into the `heat` buffer through the normal ray path. When the conduction pass
(§2) lands, fire heating distant solids by radiation comes for free — no
fire-specific heat code.


## 6. The intended emergent payoffs

The layers above are designed so that the interesting behaviour *emerges*
rather than being scripted:

- **Radiation ignites things.** A beam or a fire radiates heat → distant wood
  crosses `ignition_temp` → it catches. No line-of-fire special case.
- **Lasers melt through walls.** Beam deposits heat → conduction carries it into
  the wall → temperature crosses the wall limit → `wall_hp` drains → breach.
- **Lasers tunnel through smoke.** Beam heat triggers smoke burn-off along its
  path, clearing a line of sight it created itself.
- **Heat travels along metal.** A hull hit spreads fast along connected metal
  until it reaches an interior wood wall and ignites it — the chain crosses a
  room with no air-temperature field involved.
- **Firestorms cascade on wind.** Wind intensifies fire → hotter fire radiates
  further → ignites more fuel; a shockwave's transient wind spike kicks the
  whole cascade off.


## Implementation status

Built, running, and honest about the seam between the three layers.

**Built (layer 3 — fire):**

- `FireSimulation` in `cpp/src/fire_simulation.{h,cpp}` implements the §5
  signed-logistic intensity feedback (grow/die with the `hot`/`o2`/`avail` gates
  + the wind fan/strip terms), the own-tile plume pressure deposit, smoke
  emission, and wall burn-through. The cellular spread (12-connected stencil,
  wind-biased spread) and the backwards O₂-consumption subtraction are
  **deleted** — spread is radiation → heat → temperature → ignition (§4). It
  returns destroyed-tile coordinates; the runner destroys them.
- `PhysicsRunner` (`src/simulation/physics_runner.py`) constructs the solver,
  binds the `[physics.fire]` parameter table from `config.toml`, and calls
  `fire.step(...)` once per tick at full `sim_time` — passing `gmap.temperature`
  (the `hot` gate), the shared `gmap.wind_x`/`wind_y` (the wind term, so
  shockwaves fan/blow fires), and `gmap.is_vacuum` — then forwards
  burned-through tiles to `gmap.destroy_wall`.
- Fire ignition by explosion is built: `apply_explosion`
  (`src/simulation/physics.py`) sets `fire` on flammable tiles inside the blast.

**Built (layer 1 plumbing — shipped, unconsumed):**

- The `heat` buffer is shipped on the GameMap as a **Q16.16 int32** field,
  written in-place, and the ray engine **does deposit into it** — the
  directional march (`march_ray_directional` in `cpp/src/raycaster.cpp`)
  quantizes and saturating-adds heat for any source with a non-zero heat
  emission, using `HEAT_SCALE`/`heat_quantize`/`heat_saturating_add` from
  `raycaster.h`. The fixed-point format (§3) is real and in use *on the deposit
  side*.
- The `conductivity` per-tile cache is built and patched by the GameMap
  (`_update_caches`, `on_tile_changed`) from the material table's
  `conductivity` column. The `ignition_temp` column exists in `config.toml` and
  `MaterialTable`.
- **`heat_atten` is consumed (§1): the ray march's independent 4th channel.**
  The directional march carries R, G, B, AND a scalar heat survival, each
  attenuated by its own material coefficient — the RGB channels by `light_atten`,
  the heat channel by `heat_atten` (air 0.0, walls 1.0, glass 0.3). The deposit
  is `src.heat · heat_survival · falloff` (quantized + saturating-added into the
  Q16.16 `heat` buffer), decoupled from the RGB survival, so heat and light
  occlusion diverge (a heat-shield blocks heat but passes light; smoked glass the
  converse). The ray marches until ALL FOUR channels are below the cull epsilon.
  The per-tile `GameMap.heat_atten` cache is the `heat_atten` column projected
  onto the grid, built/patched through the same seam as `light_atten`.
- The `EnvironmentProfile` (`src/simulation/environment.py`) carries
  `temperature_min`/`temperature_max` and `environmental_damage_rate`.

**Designed, not built (layers 1–2 — the consumers):**

- **No `temperature` field exists.** There is no `gmap.temperature` array and no
  conduction/relaxation pass. The `heat` buffer is deposited into but **read by
  nobody** — it is cleared (intended) at cleanup and never consumed. The §2
  relaxation scheme is design only.
- **No ignition-by-temperature.** Fire today starts only from explosions; the
  `temperature ≥ ignition_temp ∧ O₂` path (§4) is unbuilt. `ignition_temp` is
  loaded but unused.
- **No thermal wall failure.** Walls are destroyed by fire burn-through and
  explosion damage only; the temperature→`wall_hp` melt path (energy-weapon
  melt-through) is unbuilt.
- **No smoke burn-off, no laser tunnels.**
- **No unit heat damage.** `EnvironmentProfile` temperature tolerance is data
  only; no tick handler samples `heat` at unit footprints or applies
  environmental damage. (This is consistent with the unit spec, which defers all
  environment-damage behaviour.)
- **Fire as a ray light source is not yet wired.** Fire deposits heat *only*
  through the legacy scalar `update_from_fire` path (which writes the scalar
  `light_map`, not the RGB/heat buffers). The §5 "fire as a normal `LightSource`
  in the directional pass" integration — and therefore fire's heat reaching the
  `heat` buffer — is designed but not connected; fire tiles are not yet added to
  the directional source list.

**Gaps / things to settle at build time:**

- The exact fixed-point width and rate set for `temperature` (the `heat`
  deposit is fixed at Q16.16; temperature can match it). Validate cross-machine
  bit-exactness before committing to lockstep; the float-temperature fallback
  (§3) stays available.
- **Resolved:** fire now reads the shared `wind_x`/`wind_y` field (= −grad of
  `atmosphere + wave_p`), so a shockwave's transient wind spike fans/blows fires
  directly. The old internal `grad(atmosphere)` (no wall boundary, read across
  walls) is gone.
- **Resolved:** fire parameters live in `config.toml` `[physics.fire]`, bound
  through `physics_runner.py` (the `FIRE_*` module constants are only fallbacks).
- The per-tick `heat` deposit must be cleared each tick once a consumer exists
  (it is a deposit buffer, not an accumulator across ticks); the temperature
  pass reads it non-destructively, then it resets.

**comments from erik**  exact fixed point - should we make a class out of it? or does it basically all ready exist?

**comments from erik** water and fire must interact obviously, i havent talked about it much since water is not yet done
water should be able to be turned to vapour
Water has 3 states in pressure, gas, fluid and solid, but in vacuum only gas and solid - let's try to get this modelled !
water turning to water vapour will cool it's tile, as in real life. this stuff probably belong in water rather than here, all though it's hard to say since it affects temperature just as much as fire in a way. Actually there is an argument that temperature should be its own chapter, and water and fire as well, but i guess it's fine as it is now as well.
Fluids really deserve it's file so in a way, it's reasonable to link fire with temp and water with fluid. even tho we'll have more fluids.