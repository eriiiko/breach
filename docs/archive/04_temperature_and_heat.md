# 04 — Temperature & Heat

_Depends on: [01 State & Ownership](01_state_and_ownership.md),
[02 Material System](02_material_system.md), [03 Ray Engine](03_ray_engine.md).
Status: DRAFT (rev.2, post-review)._

How heat becomes temperature, spreads, ignites things, and stays deterministic.
(Reconciliation: C7, C8. Review items: #4, #5, #6, determinism gaps.)

## Where heat comes from, where temperature lives (C7)

- **Heat is deposited by rays** (ch.03) into the `heat` buffer — fire, energy weapons,
  explosions. **Heat crosses air as radiation (rays), not an air-temperature field.**
- **Temperature lives on solids only**, implemented as a **dense full-grid field** with
  **conductivity = 0 on air** (air tiles are no-ops at ambient). Dense layout = GPU-friendly, no
  sparse gather; full-grid structure leaves the future temperature→pressure coupling a cheap add.

## Conduction scheme — faked, unconditionally-stable (C7, review #4)

**Decided: the conduction is a non-physical, unconditionally-stable relaxation** (per-tick
relaxation toward a conductivity-weighted neighbour blend, **power-of-two rates**), *not* a
CFL-limited physical diffusion. This is a visual/gameplay system, not thermodynamics, and the
relaxation:

- runs **one pass per tick** — **no substep loop** (the rev.1 "~17 CFL substeps" model is
  dropped from the locked design);
- needs **no harmonic-mean division** — power-of-two rates make the fixed-point update
  **shifts + adds** (exact, deterministic);
- removes the CFL dependence on `κ`.

The only invariant to preserve: heat spreads along high-conductivity material and **accumulates
to the ignition threshold**.

## Determinism — fixed-point, for cross-machine + multiplayer (C8, review #5)

> **Principle: fixed-point integers where a value crosses a discrete threshold into sim state;
> float where it stays continuous and perceptual (rendering).**

**Decided: keep fixed-point for `heat` AND `temperature`** — Erik wants the option of **lockstep
multiplayer**, which requires **cross-machine bit-identical** simulation (machines exchange only
inputs and each runs the identical sim; float results would desync). Replays inherit the same
benefit. The relaxation scheme makes fixed-point temperature nearly free, so the cost is gone.

Two *distinct* determinism claims (review separates them):

1. **`heat` deposit → int `atomicAdd`** — many rays → one cell; integer addition is
   order-independent → deterministic, cross-machine. (Sound.)
2. **`temperature` field → fixed-point relaxation** — a **gather** stencil (no atomics), so its
   determinism rests on **fixed rounding of the relaxation update**, *not* on atomics. With
   power-of-two rates the update is exact shifts → bit-identical cross-machine. **Validate with a
   cross-machine bit-exactness test before locking;** fallback if ever needed = float temperature
   + int heat-deposit.

**Render-only channels** (`light_rgb`, `smoke_glow`, `light_dir`) stay **float** — no downstream
threshold (stealth is image-based, ch.03 #1), and fixed-point would band the near-dark + fight
HDR.

### Fixed-point format (review #6, gap)

- **Scale:** Q16.16 (or a chosen integer scale) — *confirm the exact width when prototyping.*
- **`atomicAdd` = saturating** (clamp at max, **never wrap**) — protects the ignition threshold
  under a firestorm where many emitters deposit into few cells.
- **Thresholds** (`ignition_temp`, etc.) are **quantized into the fixed-point domain once at
  load**, with a pinned rounding mode; the comparison is `temperature ≥ quantized_ignition_temp`.
  This is the single most determinism-critical conversion — it is fixed, not per-tick.

## What temperature drives (field reactions — these mutate the world)

Separate from the read-only ray kernel; they consume the buffers and run per the ch.03 tick order
(step 8):

- **Ignition:** `temperature ≥ ignition_temp` (fixed-point) **and** O₂ present → start fire,
  consume O₂, emit smoke. *(The O₂ gate remains a float `atmosphere` threshold — single-machine
  deterministic today; if cross-machine ignition matters it shares the float-fallback exposure.)*
- **Wall thermal failure:** `wall_hp` (GPU-resident, GPU-written, ch.01) depletes as temperature
  crosses a material threshold → `destroy_wall` delta. This is how energy weapons "melt through"
  walls. Atomic-free stencil → deterministic.
- **Smoke burn-off (laser tunnels):** a field system removes smoke where heat is high (the laser
  deposits heat as a ray; the field reacts — keeps the ray world read-only).

## Emergent payoffs (intended)

- **Radiation ignites things:** a beam/fire radiates heat → distant wood crosses `ignition_temp`
  → fire, no scripting.
- **Firestorm:** wind intensifies fire → hotter fire radiates further → ignites more; a
  shockwave's wind spike cascades. (Radiation-driven — reinforces solids-only temperature.)

## Current code (where this lands)

- New `gmap.temperature` (fixed-point int) + a per-tile `conductivity` cache (from the table,
  ch.02). Neither exists today.
- A temperature/heat solver in the physics step (the dense full-grid **relaxation** + the field
  reactions above). `LightSource.heat` finally feeds it.

## Open / deferred

- **Hot-tile emission** (tiles above a `glow_threshold` become blackbody ray-emitters: RGB by
  temperature + heat) — **own chapter**; adds the `emissivity` column. Additive; nothing here
  depends on it.
- **Temperature → pressure coupling** (thermal-expansion firestorm) — needs air temperature; the
  full-grid structure keeps it a cheap later add.
- Exact fixed-point width + the cross-machine bit-exactness validation — at prototype time.
