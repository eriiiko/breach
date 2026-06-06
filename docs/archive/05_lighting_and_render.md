# 05 — Lighting & Render

_Depends on: [03 Ray Engine](03_ray_engine.md). Status: DRAFT (rev.2, post-review)._

The render side: turning the ray engine's **summed** buffers into lit pixels. Everything here is
**render-only** — zero gameplay effect, never part of the deterministic sim. (Review items:
#7, #8/#9, #16, #17, #18.)

## The interface = the summed buffers, nothing more

The ray engine (ch.03, simulation) **accumulates every source** into a tiny summed set; the
renderer reads only the sums and produces the final pixel. It never sees sources, rays, or
per-direction distributions:

- **`light_rgb`** — total light colour reaching the tile (sum of all sources)
- **`light_dir`** (dx, dy) — one aggregate light direction (for normal-map relief)
- **`smoke_glow`** — total in-smoke glow (**RGB**, ch.03 #9)

This compact summed contract is the entire sim→render seam; it's fixed-size regardless of light
count (review, your point). `heat` is sim-only and the renderer ignores it (until the deferred
temperature→glow feature).

## Render buffer packing (review #7, blocker)

Six→eight render channels no longer fit one RGBA8 texture. Pack into **two `RGBA16F`** textures
(16-bit float chosen to avoid near-dark banding at ambient ~0.01 and to carry HDR for tone-map):

- **Texture A (RGBA16F):** `light_rgb` (RGB) + `light_dir.x` (A)
- **Texture B (RGBA16F):** `smoke_glow` (RGB) + `light_dir.y` (A)

`LightingPass` becomes a pure consumer: read GameMap's **f32 accumulation** buffers →
**down-convert** + pack into the two RGBA16F textures → upload → shader. The f32→16F conversion at
pack time is the only place precision narrows (ch.03 "accumulation ≠ storage"). The shader
reconstructs `light_dir = vec2(texA.a, texB.a)` (it samples **both** textures). With 16F, store
`light_dir` **signed** — drop the old `0.5 + 0.5·x` pack and the `(sample−0.5)·2` decode. No
raycast, no duplicate light buffer (C4).

## The lighting shader

Per render pixel:

1. Sample `light_rgb` → incoming light colour/intensity.
2. **Diffuse:** `albedo × incoming_RGB` (a red lamp lights surfaces red).
3. **Normal mapping:** reconstruct `L = normalize(vec3(light_dir.xy, u_light_z))` and dot with
   the per-pixel normal. The `u_light_z` uniform fakes the out-of-plane component (existing
   trick — folded into the spec here). *The normal-map interaction is in the shader, not the ray
   march.*
4. **Tone-mapping (review #17): ACES** filmic on the linear RGB (keeps colours punchy without
   white-clipping), then sRGB encode. Add ambient floor + vacuum-tile discard (existing).
   *(ACES is render-only and swappable; A/B in the lighting demo.)*

**Combined (the load-bearing formula):** `lit = ACES(albedo × (ambient + incoming_RGB × ndotl)); out = sRGB(lit)`.

**Multi-light limitation (review #16):** `light_dir` is a single intensity-weighted aggregate, so
opposing lights average toward flat relief and per-light coloured relief isn't recoverable. This
is an accepted single-dominant-direction approximation — documented, not a bug.

## God-rays / lit smoke (review #8)

`smoke_glow` (RGB) is the deposited, smoke-absorbed light — drawn as the volumetric shaft inside
smoke. It **supersedes** the old surface-tint `light_modulation` in `overlays.py` (one
energy-conserving mechanism, no double-count). Draw order: the **additive** glow overlay is
composited **with the smoke overlay, before units/foreground**, so units drawn afterward occlude
it in screen space; and since the ray deposits no glow past opaque tiles, shafts already terminate
at walls.

## Blend discipline (review #3 — scoped)

- **Alpha-blended ("over") overlays MUST use premultiplied alpha + `BLEND_ALPHA_PREMULTIPLY`**
  (smoke surface, pressure) — Raylib's default `BLEND_ALPHA` reduces destination alpha and bleeds
  the background through opaque pixels (the "galaxies through the ship" bug).
- **Additive overlays (fire, god-ray glow) use `BLEND_ADDITIVE`** — additive raises RGB without
  touching destination alpha, so it's *also* safe and must **not** be premultiplied. The premul
  rule is scoped to alpha-blend only.

The final RT→screen blit stays premultiplied. See `renderer/overlays.py`,
`renderer/pressure_overlay.py`, `renderer/world_composite.py`.

## Specular & transmissive materials

- **Specular** (hull glint, water sheen) = **normal + specular maps driven by the directional
  light field's intensity**, not real ray bounces (consistent with no in-kernel reflection,
  ch.03 C13). Note specular adds energy on top of diffuse — the ACES stage handles the overbright.
- **Transmissive materials** (glass/tinted window): the light model tints what is *behind* the
  glass in the light field, but the diffuse layer is still flat per-tile art. v1: you see a lit
  glass pane and the tint shows on surfaces beyond it (see-through is not faked). Note it so it's
  not an art surprise.

## Refresh cadence (review #18)

The cast now runs at the **sim tick rate** (it moved into the sim, ch.03), which is lower than
render frame rate. State the cadence; if lamp motion / fire flicker visibly steps, add render-side
interpolation between sim snapshots. The render snapshot may be **one tick stale** (invisible);
the **sim never reads stale** (ch.01 freshness split).

## Headless

No renderer → the visual buffers (`light_rgb`, `light_dir`, `smoke_glow`) are **skipped/unallocated**
(ch.01 conditional buffers); the sim computes only `heat`. The skipped channels are **pure
sinks** — they must not feed back into `heat`/`temperature` (confirm), so headless and rendered
runs produce identical sim trajectories.

## Current code (where this lands)

- `shaders/lighting.fs` — scalar → `albedo × incoming_RGB`; keep the normal-map dot (+`u_light_z`),
  add the **ACES** stage before sRGB, keep vacuum discard.
- `renderer/lighting.py` — strip the raycast; become read→pack(2×RGBA16F)→upload.
- `renderer/overlays.py` / `world_composite.py` — already premultiplied; god-ray glow joins the
  **additive** path; retire `light_modulation` (superseded by `smoke_glow`).

## Open / deferred

- **Hot-tile blackbody glow** (render side of the deferred emission feature) — grey→orange→white
  colour-lerp; pairs with the ch.04 emission chapter.
- **Smoke normal maps** (internal smoke shading) — parked; revisit after lit-smoke + god-rays land.
