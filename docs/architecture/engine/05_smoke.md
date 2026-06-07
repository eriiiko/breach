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
and smoke; glass blocks smoke yet passes light). The solver now reads a per-cell **permeability**
field (`gmap.dyn_permeability`, rebuilt each tick in `stamp_units`), gathering flux across each face
as `face = min(perm[self], perm[neighbor])`: walls are `0` (sealed), and a living unit is **soft** —
a partial value (default 0.5, `[physics] unit_permeability`) so smoke seeps *past* a body instead of
reflecting off it. Behaviour is identical to the old boolean `obstacles` boundary for the current
materials. Walkability is a different predicate again and is irrelevant to smoke.

Vacuum tiles are a hard sink: smoke on any `is_vacuum` tile is zeroed each step (interior smoke
clamped to `[0, 1]`), and advection is skipped on impermeable and vacuum tiles (they would read
garbage gradients). Zeroing vacuum is what makes a breach drain a room — smoke advects toward the
breach and is deleted there, exactly as venting to space should look.

**Lingering-smoke venting (owed).** With the current vacuum-*relaxation* drain, the wind that carries
smoke out dies as interior pressure approaches zero (the gradient vanishes), so a ship breached to
vacuum keeps a stubborn haze long after the air is gone — it looks wrong. The atmosphere chapter's
**face-flux drain** (ch.04) was the intended fix, but face-flux *as a pressure sink* was attempted
and reverted: with `d_atm = 200` it cannot clear a vented room (diffusion flattens the interior
gradient → wind → 0 → smoke still lingers). The real fix needs a *sustained continuity wind toward
the breach*, which is an open atmosphere-side design decision (§4 there). (`dt_scale` and
`advection_rate` already let smoke move faster than a literal wind reading for feel; that is tuning,
not the root fix.)

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

### 6.1 Normal-mapped smoke — render model

Smoke density is one value per physics tile, but the *visual* smoke pixels within a tile sit at
render resolution. A per-pixel "smoke normal" sub-texture lets smoke not just attenuate light but
catch directional highlights from it, using the same per-pixel dot-product the renderer already does
for solid sprites (the light direction is in `gmap.light_dir`). The result is internal shading
inside the volume — wisps and eddies reading as structure instead of flat fog. The original seed —
that the normal need not be authored but can be *generated* from **wind** (a dynamic tilt that makes
smoke read as moving) plus **∇smoke** (the density gradient, the cloud's true silhouette/edge) — was
Erik's idea (2026-06); the model below is the worked-out spec built on it. **This is a design spec,
not yet built** (no shader path exists; see Implementation status).

Design target: 2D top-down, real-time, coarse per-tile inputs (`smoke_density`, `wind`, optional
`temperature`), per-pixel `light_dir` already available from the normal-map shading path, and a
per-gas per-channel `ABSORPTION` triple (§6.2). This is the richest reasonable version; **every block
is independently disable-able** for performance scaling (see tiers at the end).

**Key principle:** density gives the silhouette and bulk; a higher-res, wind-advected noise layer
gives the wisps; they live at different resolutions and are combined per-pixel. Never let the coarse
tile grid be visible — all per-pixel richness comes from the advected noise.

#### Inputs and resolutions

| Field | Resolution | Source |
|---|---|---|
| `D = smoke_density` | per-tile, bilinear → per-pixel | atmosphere solver |
| `W = wind (wx,wy)` | per-tile, bilinear | atmosphere solver |
| `T = temperature` | per-tile, bilinear | heat buffer (commit `6d3cc22`) |
| `ABS = (aR,aG,aB)` | per-gas constant | gas material (`[gases.*]`, §6.2) |
| `light_dir` | per-pixel (x,y,z) | existing normal-map shading path |
| `h` detail height | ≥ 4× density res | procedural fbm/curl, advected (step 2) |

**1. Per-pixel smoke NORMAL (the core).** Build a screen-space normal `N` (x right, y down, z toward
viewer) by summing four contributions, then normalizing.

*1a. Density-gradient normal (silhouette / bulk shape).* Central differences on the smoothed density
field — makes cloud *edges* catch light like a rounded shoulder.
```
gD       = vec2( D(x+e)-D(x-e), D(y+e)-D(y-e) ) / (2e)   // e = 1 tile
n_dens.xy = -gD                  // surface tilts away from increasing density -> lit rim
```

*1b. Wind orientation tilt (motion shaping).* A small constant surface tilt along wind so moving
smoke reads as flowing, and highlights bias to the windward face:
```
n_wind.xy = k_wind * W           // k_wind ~ 0.15..0.3
```

*1c. High-res animated noise normal (wisps — the money layer).* Gradient of the advected detail
height `h` (step 2), sampled far finer than the tile grid:
```
gh         = vec2( h(u+du)-h(u-du), h(v+dv)-h(v-dv) ) / (2du)
n_noise.xy = -k_noise * gh        // k_noise ~ 0.6..1.2
k_noise   *= mix(1.2, 0.5, saturate(D))   // thin smoke wispier, dense smoke smoother
```

*1d. Z-component (puffiness).* Denser/higher regions bulge toward camera; never let z→0 (avoids flat
black normals):
```
nz = mix(0.4, 1.0, saturate(D)) + 0.3*h
```

*Combine and normalize* (weights are the main artistic knobs — typical: density 1.0, wind 0.2, noise
0.8):
```
N.xy = n_dens.xy + n_wind.xy + n_noise.xy
N.z  = nz
N    = normalize(N)
```

**2. Flow-map / curl-noise advection of the detail texture.** The detail height `h` must roil and
travel along the wind, or it looks like a static screen-door over moving smoke. Use curl-noise for
the velocity field and a two-phase flow-map blend for the texture sampling.

*2a. Curl-noise velocity (divergence-free roiling — Bridson 2007).* From a scalar fbm potential `ψ`,
the 2D divergence-free velocity is the perpendicular gradient. Divergence-free ⇒ no sources/sinks ⇒
swirling, mass-conserving look. Wind is the bulk drift; curl is turbulence on top:
```
v_curl = ( ∂ψ/∂y , -∂ψ/∂x )       // discrete central diff, small ε
flow   = W + k_curl * v_curl       // k_curl ~ 0.3*|W| + small const
// 2-3 fbm octaves in psi; higher octaves -> smaller vortices / finer wisps
```

*2b. Two-phase flow-map advection (Valve / Catlike Coding) to hide UV stretch.* Advecting UVs by
`flow*t` stretches unboundedly; reset with two half-period-offset phases and a triangle-wave
crossfade so the seam is never visible:
```
prog_A = frac(t*speed)
prog_B = frac(t*speed + 0.5)
uvA    = uv_base - flow*prog_A
uvB    = uv_base - flow*prog_B
hA     = fbm(uvA * detail_scale)
hB     = fbm(uvB * detail_scale)
wA     = 1 - abs(1 - 2*prog_A)     // triangle: 0 at reset, 1 mid-phase
wB     = 1 - abs(1 - 2*prog_B)     // wA + wB == 1
h      = hA*wA + hB*wB
// anti-pulse (Vlachos): t += noise(uv)*0.3 so pixels don't reset in lockstep
// speed scales with |W| -> faster wind = faster roil
```
This single advected `h` feeds both the noise normal (1c) and the density detail break-up (step 5) —
one sample set, reused.

**3. Cheap self-shadowing / internal lighting (normal·light_dir).** With per-pixel `light_dir` and
`N`, do Lambert + a wrap term to fake the subsurface/multi-scatter glow real volumetric smoke has on
its lit side:
```
ndl   = dot(N, light_dir)
lit   = saturate(ndl)                       // direct lit face
wrap  = saturate((ndl + w) / (1 + w))       // w ~ 0.5, soft wrap = cheap scatter
shade = mix(ambient, 1.0, lerp(wrap, lit, 0.5))   // shade in ~[ambient, 1]
```
This single per-pixel normal stands in for the AAA "6-way lighting" idea (6 directional
density-occlusion maps blended by `dot` with the light) — the right tradeoff at 2D tile resolution.

Add a coarse **directional self-shadow** by marching density a few tiles toward the light (tile-res,
not per-pixel — Beer–Lambert along the light):
```
occ = 0
for s in 1..3:  occ += D( pos + light_dir.xy * s*tile )
selfshadow = exp(-k_sh * occ)
shade *= selfshadow
```
3 taps is plenty: the per-pixel normal carries fine relief, this carries the bulk "back of the cloud
is dark" cue.

**4. Black-body EMISSION (hot smoke glows).** Hot smoke/fire is *additive* and must not be attenuated
like cold smoke. Drive emission from `T` (heat buffer) through the black-body curve — the same curve
as the `[gases.*]` sub-note (§6.2), expressed here as a real-time fit.

*4a. Temperature → RGB (cheap polynomial fit, Kelvin).* Anchors: ~1000 K deep red, ~1900 K orange,
~4500 K yellow-white, ~6500 K white.
```
bb(T):   // T in Kelvin
  r = saturate( 1.0 )                              // red saturates early
  g = saturate( 0.39*ln(T/100) - 0.63 )
  b = saturate( 0.543*ln(T/100 - 10) - 1.196 )     // ~0 below 2000K
emit_color = bb(T)
```
(For artist-controlled exactness, sample the LUT in the `[gases.*]` sub-note (§6.2) instead of this
fit — same shape.)

*4b. Intensity (tunable ramp, threshold + power).* Don't use raw T⁴ — it clips instantly; use a
tunable power and threshold so only genuinely hot tiles glow:
```
e        = pow( saturate((T - T_glow0) / (T_max - T_glow0)), p )   // T_glow0 ~ 600K, p ~ 2..4
emission = glow_gain * e * emit_color
```

*4c. Fold in additively, post-attenuation:*
```
out_rgb += emission        // ADD, not multiply — emission ignores ABS
```
Reuses the existing `smoke_glow` buffer; `T` already exists from the heat pass.

**5. The `smoke^gamma` contrast trick.** Raw density-opacity looks like flat fog. Remapping density
through a power curve restores high-contrast wispy edges that read as "smoke," not "haze." Apply to
the **opacity / detail-modulation**, not the light:
```
D_eff = pow( saturate( D * (0.5 + 0.5*h) ), gamma )   // gamma ~ 1.5..2.5
gamma = mix(2.5, 1.2, saturate(D))                    // thin wisps sharp, dense walls soft
```
- `gamma > 1` crushes thin smoke toward transparent and sharpens edges (wispy, filmic).
- Modulating `D` by detail height `h` *before* the power breaks the coarse tile silhouette into
  filaments — this is what kills the visible tile grid.

**6. Decoupled, SCALABLE per-channel absorption vs glow gain.** The crux for Breach's gameplay (a
beam must travel *far* through coloured smoke and the smoke must still glow). **Absorption and glow
are independent and must not sum to 1.** This is the model that resolves the per-channel-attenuation
gap the shipped ray march has (see Implementation status).

*6a. Per-channel transmission (Beer–Lambert, scalable).* Exponential attenuation with a global
scalar so you can dial *how far light reaches* without changing the gas's hue:
```
tau_c   = ABS_c * D_eff * absorb_scale     // c in {R,G,B}; absorb_scale << 1 -> long reach
trans_c = exp( -tau_c )                     // never reaches zero -> beam survives deep smoke
```
`exp(-x)` is the physically correct Beer–Lambert law — the difference between "beam dies in 2 tiles"
and "beam visibly tints across the whole room." Green poison `ABS=(0.45,0.10,0.80)` ⇒ `trans≈(mid,
high, low)` ⇒ passes/tints yellow-green.

*6b. Glow gain is a separate, larger budget.* The light the smoke *scatters back* (internal lighting
step 3) uses its own gain, independent of absorption — can be > 1 while absorption is ≪ 1
simultaneously:
```
scatter_c = glow_gain_c * shade * inscatter_color_c * D_eff
```
This decoupling is what lets you author "barely absorbs, glows brightly" gases (and is exactly how
steam works: tiny absorption, large additive scatter).

*6c. Final per-channel composite* (per channel c, incoming light `L_in`):
```
L_out_c =  L_in_c * trans_c     // 1. light passing THROUGH (tinted, long reach)
        +  scatter_c             // 2. light scattered toward viewer by lit smoke
        +  emission_c            // 3. black-body self-emission (hot smoke)
```
Order matters: transmit first (multiplicative tint), then add scatter and emission (additive,
unaffected by the gas's own absorption).

#### Putting it together — per-pixel pseudo-shader

```
// --- sample coarse fields (bilinear) ---
D = sample(density); W = sample(wind); T = sample(temp);

// --- advected detail (step 2) ---
flow = W + k_curl * curl(psi, uv);
h    = twophase_fbm(uv, flow, t);              // reused below

// --- normal (step 1) ---
n_dens  = -grad(D);
n_wind  =  k_wind * W;
n_noise = -k_noise * mix(1.2,0.5,D) * grad(h);
N = normalize( vec3(n_dens + n_wind + n_noise, mix(0.4,1.0,D) + 0.3*h) );

// --- lighting (step 3) ---
ndl   = dot(N, light_dir);
shade = mix(ambient, 1.0, saturate((ndl+0.5)/1.5)) * selfshadow_3tap(D, light_dir);

// --- opacity / contrast (step 5) ---
gamma = mix(2.5, 1.2, D);
D_eff = pow( saturate(D*(0.5+0.5*h)), gamma );

// --- emission (step 4) ---
e        = pow(saturate((T - T_glow0)/(T_max - T_glow0)), p);
emission = glow_gain * e * blackbody(T);

// --- per-channel composite (step 6) ---
for c in R,G,B:
   trans_c   = exp( -ABS[c] * D_eff * absorb_scale );
   scatter_c = glow_gain_c * shade * inscatter[c] * D_eff;
   L_out[c]  = L_in[c]*trans_c + scatter_c + emission[c];
```

#### Tuning cheat-sheet (starting numbers)

| Knob | Start | Effect |
|---|---|---|
| `k_wind` | 0.2 | wisp lean / motion read |
| `k_noise` | 0.8 | wisp relief strength |
| `k_curl` | 0.3·\|W\| | turbulence vs bulk drift |
| `detail_scale` | 4–8× tile | wisp fineness (fbm octaves 2–3) |
| `gamma` | 1.2 (dense) → 2.5 (thin) | edge contrast |
| `absorb_scale` | 0.1–0.4 | **beam reach** (low = far) |
| `glow_gain_c` | 1.0–3.0 | scatter/glow brightness (decoupled) |
| `T_glow0 / p` | 600 K / 3 | when/how sharply hot smoke ignites |
| wrap `w` | 0.5 | cheap subsurface softness |
| selfshadow taps | 3 @ tile-res | bulk dark-side |

#### Scaling tiers (drop blocks for perf)

- **Tier 0 (cheapest):** normal 1a+1d, Lambert (3), contrast (5), exp transmission (6). No noise, no
  advection.
- **Tier 1:** add advected noise normal 1c+step 2 (wisps) and emission (4). — *the visual sweet
  spot.*
- **Tier 2 (richest):** add wind tilt 1b, wrap + 3-tap self-shadow (3), separate per-channel scatter
  6b, curl-noise turbulence. Full filmic look.

#### Why these choices (physics notes)

- **`exp(-τ)` over `(1-a)`** is the actual Beer–Lambert law — the difference between a beam dying in 2
  tiles and visibly tinting across the whole room. Essential for Breach's gameplay beams.
- **Divergence-free curl noise** (Bridson) makes procedural smoke roil convincingly instead of just
  scrolling — no fake sources/sinks.
- **Two-phase flow blend** (Valve) is the standard cheap way to advect a tiling detail texture
  indefinitely without visible UV stretch — 2 samples + a triangle crossfade.
- **Single-normal stand-in for 6-way lighting**: full 6-directional lightmaps are overkill at tile
  density; one advected per-pixel normal + a 3-tap bulk shadow captures the same lit-face / dark-face
  / rim cues at a fraction of the cost.
- **Additive emission post-attenuation** is physically right: a gas's self-emission and back-scatter
  are not subject to its own front-face absorption in the thin-slab approximation rendered here.
- **2D top-down ⇒ no buoyancy term**: gas motion comes purely from diffusion + wind/pressure
  advection in the solver, never from a vertical rise; "puffiness" (1d) is a *lighting* cue only, not
  physical lift.

### 6.2 Multi-gas system — `[gases.*]` material table

Smoke generalises from one scalar field to a small **set of gas density fields** — one per gas type —
sharing the *same* diffusion + advection solver (they all ride the same wind; on CUDA it is one
batched stencil). A gas type is **data-driven, exactly like a material**: a `[gases.*]` config table,
one row per gas. White and black smoke are a **confirmed requirement** (Erik): white vapour from
water/steam, black soot from fire and explosions, blending to grey through the optical model below.
Because a gas type is just a data row, further variants stay free additions — more rows, not more
system. This unifies what were separate forward ideas (a fuel field, a teargas field) into one system
and reuses 100 % of the smoke transport plus the per-channel attenuation machinery. **Design-only,
not built** — smoke is one scalar field today (see Implementation status).

The two gameplay/structural properties that ride alongside the optics:

- **Effect** is a per-gas gameplay tag, read **unit-side** (a mechanics-chapter concern, not the
  solver's): `poison` = damage-over-time to a unit in the cell; `teargas` = slow / blind / area
  denial; `white_smoke` = pure vision block. The solver only transports the field; the cell-occupancy
  reading of it lives in mechanics.
- **Flammability** is the `fuel_gas` row's defining flag: it ignites where `fuel_gas > threshold`
  with oxygen/heat present, spawning fire (heat + `black_smoke`). A **flamethrower** is then just
  *emit `fuel_gas` in a cone* (directed injection — the momentum-at-nozzle idea from the atmosphere
  chapter) *and ignite it*. That may grow into its own system owning ignition/spread; the gas is the
  combustible **substrate** it burns, and the `fuel_gas` `diffusion` is the knob between a tight
  flamethrower jet and a diffuse explosive fog. `fuel_gas` is the only flammable gas.

**Mixing falls out for free.** When several gases share a cell, the cell's optical signature is the
**density-weighted sum** of the per-gas per-channel signatures — poison over `fuel_gas` blends to a
murky olive in both the look *and* the light-tint, through the same per-channel attenuation combine
`dyn_light_atten` already performs. No blend code. **v1 keeps the gases independent**: they coexist
and blend *visually*, each with its own effect, but do not react with one another — chemical
interaction between gases is a later layer. Cost is ≈ N× the (cheap, batchable) smoke solver.

This table defines the physical/optical parameters of every gas. Each gas is a per-channel
**absorption** triple (Beer–Lambert, applied multiplicatively to the light field per §6.1 step 6),
plus diffusion/decay rates, an optional glow term, and the gameplay flags above. **All values are
design defaults — tune them in play.** RGB triples are *per-unit-density absorption*: higher = more
of that channel removed, so the gas tints transmitted light toward the channels it does *not* absorb.

> Two conventions used below:
> - **Absorption** is the *subtractive* term: `trans_c = exp(-ABS_c · D · absorb_scale)` (render model
>   §6.1 step 6). It never reaches zero, so a bright beam survives through deep smoke.
> - **Scatter/glow** is a *separate, additive* budget (not constrained to `absorb + glow = 1`). Steam
>   in particular is dominated by additive brightening, not absorption — see its `scatter_albedo`
>   note.

#### Config-shaped spec (TOML-ish)

```toml
# Per-channel ABSORPTION is (R, G, B), per unit density.
# diffusion / decay are per-tick rates (atmosphere-solver units), design defaults.
# glow is the baseline self-glow gain (separate from black-body emission, §6.1 step 4).

[gases.white_smoke]            # water vapour / steam
absorption   = [0.10, 0.10, 0.10]   # flat, low — Mie scattering is spectrally neutral
scatter_albedo = [0.92, 0.92, 0.95] # NEAR-WHITE additive brighten — the dominant visual term
diffusion    = 0.18                 # spreads readily, light gas
decay        = 0.020                # condenses / dissipates moderately fast
glow         = 0.0                  # no self-glow (but scatters local light brightly)
flammable    = false
effect       = "vision_block"       # pure concealment; clears on decompression
# NOTE: steam READS as bright white because scatter_albedo * local_light dominates.
#       Absorption alone would make it look like dilute grey smoke — do not raise it.

[gases.black_smoke]            # combustion soot
absorption   = [0.88, 0.90, 0.93]   # near-neutral, slight blue tilt: thin soot reads warm/brown
scatter_albedo = [0.04, 0.04, 0.04] # soot barely scatters — it is the dark gas
diffusion    = 0.10                 # heavier, clings; slower spread than steam
decay        = 0.008                # lingers — soot settles slowly
glow         = 0.0                  # cold soot does NOT glow; see black-body sub-note for hot soot
flammable    = false                # the soot itself is spent fuel
emits_when_hot = true               # black-body emission driven by heat buffer (see sub-note)
effect       = "vision_block_heavy" # near-opaque concealment at density

[gases.poison]                # chlorine (Cl2), yellow-green
absorption   = [0.45, 0.10, 0.80]   # B absorbed hardest, R moderate, G passes -> yellow-green tint
scatter_albedo = [0.10, 0.30, 0.06] # faint green inscatter
diffusion    = 0.12                 # creeps and pools (heavier-than-air feel)
decay        = 0.004                # persistent — the hazard lingers
glow         = 0.0
flammable    = false
effect       = "damage_over_time"   # lethal cloud; saturates to iconic green at mid-high density
# NOTE: scale opacity with density so thin wisps read near-clear and only thick
#       columns saturate to the WW1 yellow-green. Tune density->opacity to put the
#       "deadly green cloud" at mid-high density, not at trace levels.

[gases.teargas]               # CS aerosol, pale near-white (DELIBERATELY ambiguous vs steam)
absorption   = [0.12, 0.16, 0.30]   # low overall, B ~2x R -> faint warm/yellow only in thick plumes
scatter_albedo = [0.88, 0.90, 0.92] # near-white scatter, like steam — sustains the ambiguity
diffusion    = 0.15                 # disperses like a fine aerosol
decay        = 0.010                # clears moderately
glow         = 0.0
flammable    = false
effect       = "area_denial"        # forces units out of the area; non-lethal
# NOTE: kept visually NEAR white_smoke ON PURPOSE (tactical ambiguity, confirmed).
#       At a glance reads as steam; only thick plumes betray the faint yellow cast.
#       For more "off" look nudge to [0.14,0.18,0.34]; to vanish into steam drop to [0.11,0.14,0.24].

[gases.fuel_gas]              # combustible vapour, faint, FLAMMABLE
absorption   = [0.08, 0.10, 0.16]   # very faint, slight blue tilt — nearly invisible haze
scatter_albedo = [0.20, 0.22, 0.28] # weak, faintly cool scatter
diffusion    = 0.22                 # GAMEPLAY KNOB: high = gassy cloud, low = tight flamethrower jet
decay        = 0.006                # lingers as an ignition hazard
glow         = 0.0
flammable    = true                 # IGNITES -> spawns heat + black_smoke; the only flammable gas
emits_when_hot = true               # while burning it glows via the black-body curve
effect       = "ignition_hazard"    # invisible-ish until lit; then a fireball
# NOTE: diffusion is the flamethrower feel knob. A dedicated flamethrower system may
#       own ignition/spread; this gas is the substrate it burns. Low diffusion = a
#       directed jet that hangs in the air; high diffusion = a diffuse explosive fog.
```

#### Readable summary

| Gas | Real-world | Absorption (R,G,B) | Diffusion | Decay | Glow | Flammable | Gameplay effect |
|---|---|---|---|---|---|---|---|
| **white_smoke** | water vapour / steam | `[0.10, 0.10, 0.10]` | 0.18 (fast) | 0.020 (fast) | scatter-bright, no self-glow | no | Vision block; clears on decompression |
| **black_smoke** | combustion soot | `[0.88, 0.90, 0.93]` | 0.10 (slow) | 0.008 (lingers) | **hot → black-body** | no | Heavy vision block; near-opaque |
| **poison** | chlorine (Cl₂) | `[0.45, 0.10, 0.80]` | 0.12 (pools) | 0.004 (persistent) | none | no | Damage-over-time; iconic green |
| **teargas** | CS aerosol | `[0.12, 0.16, 0.30]` | 0.15 | 0.010 | scatter-bright (ambiguous) | no | Area denial; *reads as steam* |
| **fuel_gas** | combustible vapour | `[0.08, 0.10, 0.16]` | 0.22 (knob) | 0.006 (lingers) | hot → black-body when lit | **yes** | Ignition hazard → fireball |

**Notes on the chosen values (research-backed):**
- **white_smoke** absorption dropped from the starting `[0.30,0.30,0.30]` to `[0.10,0.10,0.10]`:
  steam is Mie-regime (droplets ≫ wavelength), so extinction is spectrally flat *and* its signature
  is **additive brightening**, not subtractive darkening. The visual weight lives in `scatter_albedo`
  coupled to the local RGB light field, not in absorption. Equal-RGB shape was correct;
  magnitude/sign were not.
- **black_smoke** nudged to `[0.88,0.90,0.93]`: soot absorption rises toward blue (`~1/λ^α`, complex
  index ≈ 1.95 + 0.79i), so thin soot transmits a little extra red and reads faintly warm/brown while
  dense soot goes near-black. The starting `[0.90,0.90,0.92]` is also defensible if a more neutral
  look is wanted.
- **poison** changed from `[0.75,0.20,0.65]` to `[0.45,0.10,0.80]`: Cl₂'s visible absorption tail
  peaks in the violet/blue, so **blue must be absorbed hardest**, red only moderately (red partially
  survives → *yellow*-green, the sickly WW1 hue), green passes. The original over-absorbed red, which
  skewed toward pure cyan-green. Push R→0.6 for colder/greener; R→0.35 for more sulfurous-yellow.
- **teargas** softened from `[0.25,0.35,0.60]` to `[0.12,0.16,0.30]`: real CS is near-white
  pyrotechnic smoke with only a faint warm cast. Low overall absorption keeps thin haze bright/white
  (scatter-dominated) so it stays ambiguous with steam; B≈2×R gives a faint yellow that only emerges
  in thick plumes. The starting value was too dark and too saturated-yellow to pass for water vapour.
- **fuel_gas** (renamed from `fuel`) kept faint and near-invisible per the approved palette; it is
  the only flammable gas and the substrate a flamethrower/ignition system burns. Its `diffusion` is
  the primary gameplay knob (tight jet ↔ explosive fog).

#### Sub-note — temperature → black-body emission (`black_smoke`, and any gas with `emits_when_hot`)

Hot soot glows. Emission is **additive and is not subject to the gas's own absorption** (thin-slab
approximation), so it is composited *after* transmission (render model §6.1 step 4 / step 6). It has
two separable parts: **chromaticity** from a temperature LUT, and **intensity** from a
Stefan–Boltzmann-style (∝ T⁴) ramp. Although `black_smoke` is the canonical emitter, *any* gas
flagged `emits_when_hot` (e.g. burning `fuel_gas`) uses the same curve — drive `T` from the heat
buffer (commit `6d3cc22`).

**Chromaticity LUT** (Planckian locus, clipped to sRGB, normalized so brightest channel = 1.0; lerp
between rows):

| Temp (K) | R | G | B | Look |
|---|---|---|---|---|
| 600  | 0.10 | 0.00 | 0.00 | barely-visible dull red (near black) |
| 800  | 0.40 | 0.05 | 0.00 | dim ember red |
| 1000 | 1.00 | 0.22 | 0.00 | deep red-orange |
| 1300 | 1.00 | 0.40 | 0.05 | orange |
| 1600 | 1.00 | 0.55 | 0.16 | bright orange |
| 2000 | 1.00 | 0.68 | 0.33 | amber / candle |
| 2500 | 1.00 | 0.78 | 0.55 | warm yellow-white |
| 3000 | 1.00 | 0.84 | 0.68 | incandescent (soft white) |
| 4000 | 1.00 | 0.92 | 0.85 | warm white |
| 5000 | 1.00 | 0.98 | 0.96 | near-neutral white |

Red saturates to 1.0 by ~1000 K and stays there; G then B climb as T rises, walking
red→orange→yellow→white. Fire/smoke never exceeds ~5000 K, so that is a fine top of the table. Store
as a small LUT and linear-interp between rows.

**Intensity + final emission** (Stefan–Boltzmann, shifted so cool smoke is genuinely dark):

```
# --- colour (chromaticity only) ---
rgb_chroma = LUT_lerp(T)                 # table above, linear interp between rows

# --- intensity (relative, Stefan-Boltzmann, shifted) ---
T0   = 800.0                             # visible-emission threshold; below this -> dark
Tref = 2000.0                            # reference where I ~= 1.0
if T <= T0:
    I = 0.0
else:
    I = ((T - T0) / (Tref - T0))**4      # T^4 law, shifted so it is dark below T0
I = min(I, I_max)                        # clamp; I_max ~ 3.0..4.0 for white-hot core

# --- final emission folded into the light field (ADDED, post-attenuation) ---
emission_rgb = rgb_chroma * I * smoke_density   # glow lives in the soot that is actually there
```

Why this shape: **T⁴** gives the correct steep ramp (cool 1000 K soot is a dim red ember; 2000 K+
blows out to yellow-white). **Shifting by `T0 = 800 K`** before the power makes cool smoke genuinely
dark (I → 0) — selling "cool smoke is just shadow." **Clamping `I_max`** lets the hottest core
saturate to white and feed the existing ACES tone-map (commit `6df05ec`), which does the
red→orange→yellow→white roll-off for free as I drives all channels past 1.0. **Multiplying by
`smoke_density`** ties glow to the soot present, so it lives in the smoke volume, not empty air. If a
separate `heat` buffer exists (commit `6d3cc22`), drive `T` from it and keep density purely as the
visibility/attenuation term. *Pure-Planck alternative:* `I = (T/Tref)**4` minus a small floor,
clamped at 0 — same behaviour, softer cutoff.

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
- **Permeability boundary + soft units** — the solver reads per-cell `dyn_permeability`
  (`face = min(perm[self], perm[neighbor])`) instead of the bare boolean; a living unit writes a
  *partial* value (default 0.5, `[physics] unit_permeability`), so smoke seeps past a body.
  Behaviour-identical to the old boolean boundary for the current materials.
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

- **Lingering-smoke venting** — the permeability boundary and soft units have landed (above), but the
  lingering-haze-on-vacuum artifact is **not** fixed. The fix is **smoke-side**, not atmosphere-side:
  the atmosphere's vacuum-relaxation drain is adequate, and adding a continuity-wind / *second wind*
  to the atmosphere is the wrong layer (resolved in ch.04 §4, which now says the same). The lingering
  haze is **smoke v2**: replace the central-difference advection stencil (which checkerboards near
  breaches) with **semi-Lagrangian advection**, plus a **dial-able smoke-side sink-pull** that biases
  smoke advection toward the nearest breach. That sink-pull is a bias inside *smoke* transport — it
  never touches the pressure field. Designed, not built.
- **Normal-mapped smoke** (§6.1) — now **fully specified** (the per-pixel normal + curl-noise
  advection + per-channel composite model in §6.1); still not built — there is no smoke-normal
  texture and no shader path.
- **Multi-gas system** (§6.2) — smoke is a single scalar field today; the N-field gas set is now
  **specified**: a data-driven `[gases.*]` table (`white_smoke / black_smoke / poison / teargas /
  fuel_gas`) with per-channel **absorption** (subtractive, Beer–Lambert) **plus a separate additive
  `scatter_albedo`**, density-weighted mixing, and `emits_when_hot` black-body emission. Not built.
  Flamethrower (`fuel_gas` + ignition) follows from it.
- **CUDA path** (§3) — semi-Lagrangian GPU advection is planned; the current solver is CPU C++. (Note
  the smoke-v2 semi-Lagrangian advection above is wanted on the CPU path first, and ports to the GPU
  pattern unchanged.)

**Gaps / known issues:**

- **Explosion-smoke noise too subtle** (§4) — confirmed in code: the `[0.4, 1.0]` per-tile
  multiplier saturates near the blast centre and diffuses out at the edges within a few substeps,
  so the cloud lacks visible texture. Needs more dramatic initial structure. A live **dial/knob** to
  tune the noise amplitude and scale in the demo (rather than editing constants and rebuilding) is
  wanted so the look can be found by eye. Open.
- **Per-channel smoke attenuation** — the design now resolves this. The multi-gas optical model
  (§6.2) splits a gas's signature into a per-channel **`absorption`** triple — subtractive,
  Beer–Lambert `exp(-τ)`, with a global `absorb_scale` so beams travel *far* through coloured smoke
  without changing its hue — **plus a separate additive `scatter_albedo`** so a gas can brighten
  (steam) without darkening. That is the colour model the shipped path lacks. The *shipped*
  `march_ray_directional` still uses a **single scalar `smoke_absorption`**: the colour of god-rays
  comes from the deposited light's colour, not from three independent smoke coefficients. Not a defect
  — the shipped model is monochrome-absorbing — but the named build gap: realising §6.2 means giving
  the ray march the per-channel `absorption` + additive `scatter_albedo`.
- **No smoke substep stability cap** — the design relies on wind-dependent diffusion to suppress
  advection oscillation rather than a hard CFL limit; large `advection_rate` or `dt_scale` values
  can still oscillate. Consistent with the design intent but worth noting for tuning.
