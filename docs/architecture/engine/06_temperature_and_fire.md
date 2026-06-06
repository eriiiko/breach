# Temperature & Fire

**Depends on:** Grid, State & Ownership, Material System, Atmosphere, Ray Engine.

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

Fire is the built, running layer. It is a cellular system on flammable tiles:
each tile carries a `fire` intensity in `[0, 1]` (0 = unlit, 1 = full blaze),
stepped once per tick by the C++ `FireSimulation`. A tick runs these stages in
order:

1. **Spread.** Burning tiles raise the intensity of neighbouring *flammable*
   tiles. The neighbourhood is 12-connected (4-orthogonal + 4 diagonal +
   2-tile-range orthogonal), so fire reaches slightly past its immediate
   neighbours and feels like it leaps gaps.
2. **Wind-biased spread.** Fire steers its spread downwind. It computes a local
   wind from the `atmosphere` gradient and boosts ignition of the downwind
   neighbour by up to 3× (and suppresses the upwind one).
3. **Wind modulates intensity.** Existing fire is fed or blown out by wind
   according to its strength *relative to* the wind:

   ```
   wind_speed = |grad(atmosphere)|
   margin     = fire − k_wind_thresh · wind_speed
   fire      += dt · k_wind_net · wind_speed · margin
   ```

   The sign of `margin` decides the outcome: a fire above the
   wind-dependent threshold is fed (burns hotter); below it, blown out. The
   crossover scales with wind, so the same gust that fans a blaze snuffs a
   guttering flame. An explosion's shockwave is a massive transient wind spike:
   it blows out small fires and flares up big ones.
4. **Growth.** Any burning tile grows toward full intensity (≈ 0.5 / s).
5. **Flammable constraint.** Fire is zeroed on non-flammable tiles — only fuel
   burns.
6. **O₂ check.** A burning tile dies if the average `atmosphere` of its
   air-side neighbours falls below `o2_threshold`. Fire suffocates in vacuum
   and in already-burnt-out rooms — the decompression-extinguishes-fire loop.
7. **O₂ consumption.** Fire draws down `atmosphere` in adjacent air tiles,
   scaled by its intensity, so a sustained blaze starves itself and its
   neighbours over time.
8. **Smoke emission.** Fire adds `smoke` to adjacent air tiles, which the smoke
   dynamics then advect on the wind.
9. **Wall burn-through.** Fire depletes `wall_hp` on the tile it burns; when a
   flammable wall reaches zero HP it is reported as destroyed. The fire step
   returns the list of `(y, x)` tiles that burned through; the per-tick runner
   calls `destroy_wall` on each (the solver itself never edits the material
   grid).

Finally `fire` and `smoke` are clamped to `[0, 1]`; `atmosphere` is left
unclamped (the atmosphere solver owns its own bounds).

### Fire parameters

Set on the C++ `FireParams` struct, bound from the Python runner:

| Parameter | Value | Meaning |
|-----------|------:|---------|
| `spread_rate`    | 0.3  | Spread rate to neighbours |
| `o2_threshold`   | 0.60 | Minimum neighbour atmosphere for survival |
| `o2_consumption` | 0.3  | Atmosphere consumed per second per unit intensity |
| `smoke_emission` | 0.8  | Smoke produced per second per unit intensity |
| `wall_damage`    | 0.4  | Wall HP lost per second per unit intensity |
| `k_wind_thresh`  | 0.5  | Fire must exceed `k_wind_thresh · wind_speed` to survive |
| `k_wind_net`     | 3.0  | Rate of the wind feed/cool effect |

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

- `FireSimulation` in `cpp/src/fire_simulation.{h,cpp}` implements every stage
  in §5: 12-connected spread, wind-biased spread, wind intensity modulation,
  growth, flammable constraint, O₂ check, O₂ consumption, smoke emission, and
  wall burn-through. It returns destroyed-tile coordinates; the runner destroys
  them.
- `PhysicsRunner` (`src/simulation/physics_runner.py`) constructs the solver,
  binds the parameter table, and calls `fire.step(...)` once per tick at full
  `sim_time`, then forwards burned-through tiles to `gmap.destroy_wall`.
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
  `conductivity` column. The `ignition_temp` and `heat_atten` columns exist in
  `config.toml` and `MaterialTable`.
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
- Fire still computes its own wind from `grad(atmosphere)` internally rather
  than reading the precomputed `wind_x`/`wind_y` field, so the shockwave
  (`wave_p`) does not yet influence fire, and the internal gradient has no
  Neumann wall boundary (it reads across walls). Folding fire onto the shared
  wind field is the intended cleanup.
- Fire parameters are hardcoded in `physics_runner.py` rather than living in
  `config.toml` alongside the other physics tunables.
- The per-tick `heat` deposit must be cleared each tick once a consumer exists
  (it is a deposit buffer, not an accumulator across ticks); the temperature
  pass reads it non-destructively, then it resets.
