# Breach Architecture — Spec Chapters

_Created: 2026-06-05 · Status: **DRAFT rev.3 — two review rounds applied; build-ready pending Erik's final sign-off.**_

This folder is the canonical architecture, **one focused file per system**, replacing the
monolithic `docs/architecture.md` (which is being retired chapter-by-chapter). Each chapter
declares the chapters it **depends on** at the top.

These chapters consolidate the **ray-engine design reconciliation** (June 2026) — 18
cross-doc conflicts resolved, recorded in `docs/ray_engine_reconciliation.md`. That tracker
plus the stray drafts (`ray_engine_design.md`, `implementation_plan_radiation_temperature.md`)
are **superseded by these chapters and will be deleted once this draft passes review** — do
not delete them yet.

## Chapters

| # | Chapter | Covers | Depends on |
|---|---------|--------|-----------|
| 01 | [State & Ownership](01_state_and_ownership.md) | who owns world state; GPU-residency model | — |
| 02 | [Material System](02_material_system.md) | the material-property table; per-channel attenuation | 01 |
| 03 | [Ray Engine](03_ray_engine.md) | the DDA raycaster: light + heat + vision + weapons | 01, 02 |
| 04 | [Temperature & Heat](04_temperature_and_heat.md) | conduction, ignition, fixed-point determinism | 01, 02, 03 |
| 05 | [Lighting & Render](05_lighting_and_render.md) | buffers → pixels; shader; normal maps | 03 |

## The spine (one paragraph)

**Python/`GameMap` owns world state behind a `gmap.<field>` interface; C++ owns the math; the
GPU owns the field *memory* when CUDA lands.** World state is **numerical arrays + a
material-property table** (no tile-objects). The **ray engine is physics** — a deposit-only
DDA marcher over a read-only world that writes buffers (`light_rgb`, `light_dir`, `heat`,
`smoke_glow`); the **renderer is a downstream consumer** of those buffers. **Determinism**
is preserved by fixed-point integers exactly where a value crosses a gameplay threshold
(heat, temperature), float everywhere render-only. Everything is shaped to port to CUDA
unchanged: one-thread-per-ray, read-only world, `atomicAdd` deposits, no in-kernel forking.

**Why a "ray engine" lives in a *simulation* spec:** in Breach you made light *physical* — rays
carry heat that ignites fire and melts walls. Once light has gameplay consequences, *computing*
it is a simulation task, so the raycaster moves out of the renderer into the sim and the renderer
becomes a pure read-only consumer of a small **summed** buffer set. Render reads sim; sim never
reads render. (Full rationale: ch.03 §"Why the raycaster is simulation".)

## Principles

- **Canon over prototype.** Prototypes (`prototypes/`, prototype-authored levels, scratch tools)
  may freely diverge from canon. But on **any** conflict between a prototype and canon — the engine
  source (`cpp/`, `src/`, `renderer/`) + these design docs — it is resolved **in canon's favour**:
  fix the prototype, never bend the engine or the docs to it. Canon changes only if canon itself is
  found wrong, and then deliberately, via its design doc.
- **Canonize bottom-up.** A system is only locked as canon once the systems it *depends on* are
  canon — design docs sit on settled ground, not shifting ground.

## Deferred (own chapters, after the core lands)

- **Hot-tile emission** (blackbody ray-emitters above a glow threshold) — adds an
  `emissivity` column to the material table; powers the wind→fire→firestorm loop.
- **Entity re-emission** (prisms / mirrors / refractive glass) — direction-changing optics
  as secondary-pass emitters (never in-kernel forking).
- **Temperature → pressure coupling** (thermal-expansion firestorms).

## Doc-debt to close at the end of this phase

- Reword `cuda_integration_plan.md` **§7** (ownership → ch.01), **§3** (scalar light → RGB +
  deposit channels), **§4** (CFL-diffusion temperature → faked relaxation) so the CUDA plan stops
  contradicting the locked chapters.
- Delete `ray_engine_reconciliation.md`, `ray_engine_design.md`,
  `implementation_plan_radiation_temperature.md`, and retire the superseded sections of the
  old `architecture.md`.
