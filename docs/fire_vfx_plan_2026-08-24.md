# Fire VFX plan — flame-structure fluid sim (2026-08-24)

*Planning session, Erik + Claude, 2026-08-24. Status: approach locked at the
discussion level; F0 design session still to run (decisions below are
revisitable there). PARKED behind the physics lid — see Sequencing.*

Inputs: `docs/Breach_Fire_Rendering_Summary.md` (technique survey, ChatGPT,
2026-08); the [Ignitement Unity breakdown](https://unity.com/blog/real-time-fluid-simulation-fire-vfx-ignitement-breakdown)
(Sørb, 2026); `docs/fire_rendering_research.md` (our July pass);
a full systems sweep of the repo (2026-08-24).

---

## 1. The gap

Fire today is rendered as a physical **blackbody emissive field** (Tier 1
`HeatFieldOverlay`), **real ray-traced fire lights** (Tier 2
`fire_lights.py`), **dirty-Planck speckle** (Tier 3), and smoke through the
gas medium with **advected detail noise riding the real wind**. That is the
bottom half of every modern fire-VFX stack — already built, already physical.

What fire does NOT have is **flame structure**: tongues, vortices, coherent
lean and plume. Noise adds texture, not structure. A fluid sim adds structure.
That is the whole scope of this arc.

## 2. The approach (locked 2026-08-24)

**A render-only 2D stable-fluids VFX sim, one-way fed by the real physics,
rendered through the machinery we already own.** Ignitement-style: the entire
sim is GLSL fragment-shader passes over ping-pong render textures (their cost:
"comparable to a few post-processing effects" at 1024²/512²). No compute
shaders, no CUDA — which matches our stack exactly (raylib GLSL 330, no
CUDA↔GL interop; every field crosses to the renderer as numpy → texture).

Layer split (identical to our existing sim/render architecture):

- **Layer A (authoritative, untouched):** the deterministic fire/atmosphere
  sim. Fire = per-tile logistic intensity `I ∈ [0,1]` on flammable solid
  tiles; wind = EOS velocity `u` in m/s; temperature, O2, soot as shipped.
- **Layer B (this arc, render-only):** VFX fields on a finer grid —
  `u_vfx (vec2)`, `reaction R`, `T_vfx`, later `smoke_vfx`. Reads dequantized
  copies, writes NOTHING back (the `fire_lights` `heat = 0.0` pin is the
  precedent). Never runs headless. Digests untouched by construction.

### Decisions locked with Erik (revisitable at F0)

1. **Domain: whole-map fixed-resolution grid**, `M` texels per tile axis as a
   quality knob (~M=8 desktop, M=4 laptop). NOT per-fire domains, NOT a
   camera-following window in v1. Rationale: current maps are small
   (48×32…128×256 tiles → ≤ ~1024×2048 texels at M=8, Ignitement-budget
   without windowing machinery); world-anchored forever; off-screen fires
   keep simulating; zoom-independent. Caveat: all current maps are **test
   maps** — real map sizes are unknown.
   **v2 upgrade path: camera-following domain** (Ignitement's actual scheme —
   fixed texel budget, domain follows/rescales with the view, entering
   regions seeded from the game fields). Trigger: real maps outgrow the
   whole-map budget, or we want more sub-texel detail at max zoom.
2. **v1 renders FLAME ONLY.** Smoke stays with `gas_medium` (canonical "the
   only smoke/gas look"). No parallel smoke path; a later beat may feed a
   near-flame fine-smoke term INTO the gas-detail layer, never beside it.
   Also dodges the `plume_k_scale = 300` soot-scale hack (dies with #6).
3. **Sequencing: parked behind the physics lid**, specifically after #12
   (fire & heat tuning — the logistic law's behavior is this system's input
   signal) and #6 (soot density scale). Then an orthogonal arm alongside
   animation and ML.

### The sim loop (per render frame, all fragment passes)

```
u_base   = upsample(wind m/s → texels/frame)          ← real wind, correct units (#51!)
u_vfx    = advect(u_vfx) ; blend toward u_base
inject   : burning tiles (I > 0) emit R, T at their OPEN faces
           (fire lives on flammable SOLIDS; the flame front is adjacent air)
           source strength = f(I, T_tile) — dynamic fire size for free
vorticity confinement on u_vfx
project  : ~20–30 Jacobi iterations → divergence-free
advect   R, T_vfx by u_vfx ; decay
```

Top-down note: there is **no in-plane buoyancy** (the plane is the floor).
Flames lean with wind in-plane; "rise" is a rendering fiction — the
pseudo-volume/height layer (F4), not a sim force.

### The render pass

`R, T_vfx → flame mask → BlackbodyRamp color (flame, emissive floor and cast
light agree by construction) → detail-noise octaves (advected_noise + the
gas_medium.fs reconstruction tricks: bicubic B-spline, domain warp, erosion)
→ composite into the world RT per the blend discipline.` Fire lights stay
exactly as shipped (steady-glow ruling: no jitter; any future flicker is a
render-side modulation of light intensity by local R, decided at F0 or later).

## 3. Reuse inventory

| Piece | Use |
|---|---|
| `renderer/blackbody.py` `BlackbodyRamp` | flame color = the same physical map as emissive + lights |
| `renderer/advected_noise.py` bakes + tick-locked phase clock | detail octaves; the clock convention |
| `shaders/gas_medium.fs` | fork for flame detail: bicubic reconstruction, domain warp, erosion |
| `renderer/core.py` RGBA16F helpers | HDR ping-pong RTs |
| `renderer/camera.py` + `coords.py` | domain math |
| `tools/lighting_demo.py` + `levels/fire_studio` + fire tools | the tuning/HUMAN-TEST harness |
| `src/simulation/recorder.py` .npz dumps | replay real fire data through the look offline |
| `cpp/src/cuda_sl_advection.cu` | structural reference for the SL gather (not code reuse) |
| `prototypes/smoke_sim.py` | our own float Stam-loop reference |

Multigrid note: the sim's MG (`cuda_mg_solve.cu`) is an algebraic pyramid
inside the pressure solve, not a multi-resolution field system — no code
reuse, and VFX-grade projection needs only Jacobi.

## 4. Constraints & traps (from the 2026-08-24 systems sweep)

- **No CUDA↔GL interop, no compute shaders** → fragment ping-pong is the
  only GPU path (and the proven one — Ignitement chose it deliberately).
- **#51:** `gas_detail.py` reads wind with legacy `-grad(P)` units; the VFX
  sim must use m/s → `v / tile_size_m / tps` texels/tick. Fix #51 first.
- **`plume_k_scale = 300`** render hack on soot density — do not seed from
  `gas[SMOKE]` until #6 retires it.
- **Render emitters never write synced state** (`heat = 0.0`, `jitter = 0.0`
  pins in fire_lights — copy the pattern).
- **Graphics budget is decoupled from sim budget** (graphics/README founding
  principle); headless/RL runs never execute any of this.
- Budget target: ~1–1.5 ms desktop at M=8; laptop tier M=4. Measured in
  lighting_demo from F1 on.

## 5. Arc shape

- **F0 — design doc + adversarial critique** (with Erik): field set, source
  law `f(I, T)`, flame-look direction (height/parallax choice for "rise"),
  M values + budgets per machine, projection iteration count, config schema
  `[render.fire_vfx]`, canonical-systems rows. Revisit locked decisions.
- **F1 — substrate:** VFX RTs + bridge (burning-face source extraction, wind
  base field in m/s), debug overlay toggle, perf harness. No look. Gate:
  budget measured; digests trivially green.
- **F2 — the mini-Stam loop** in GLSL: SL advect → inject → vorticity
  confinement → Jacobi project. Gates: stability (F5 reload, resize,
  long-run NaN watch), budget.
- **F3 — flame render pass:** R → mask → blackbody → detail octaves →
  composite. **HUMAN-TEST (Erik, fire_studio).**
- **F4 — pseudo-volume height + embers v1** (first particle system: CPU
  particles sampling real wind + curl noise, blackbody-cooled; designed as
  the canonical particle primitive — #34/#40 will reuse it) + smoke-handoff
  polish. **HUMAN-TEST.**

Citations to carry in implementation headers (+ PDFs to `docs/papers/`):
Stam, *Stable Fluids* (1999) / *Real-Time Fluid Dynamics for Games* (2003);
Harris, *Fast Fluid Dynamics Simulation on the GPU* (GPU Gems ch. 38);
Fedkiw, Stam & Jensen, *Visual Simulation of Smoke* (2001) — vorticity
confinement; Bridson, Hourihan & Nordenstam, *Curl-Noise for Procedural
Fluid Flow* (2007); the Ignitement breakdown URL above.

## 6. Open questions for F0

1. How "rise" renders in top-down: parallax layers (screen-up vs radial
   from camera center) vs intensity/width modulation only.
2. Does `T_vfx` exist as a sim field, or is flame color sampled from the
   game temperature grid directly (cheaper, coarser)?
3. Ember counts / particle-system scope — how much of the canonical
   particle primitive lands in F4 vs its own later arc.
4. Flicker: keep lights steady (current ruling) or modulate by local R.
5. M defaults + the laptop tier; F-toggle and config schema.

## Systems

**Existing canonical systems this arc must use** (from CLAUDE.md): R1/R2
GameRenderer + WorldComposite (one new module + one `compose_world` draw
call); R3 camera/coords; R5 gas_medium remains THE smoke look (this arc does
not touch smoke in v1); R6 Blackbody (never a second temperature ramp); R13
dequantize convention; config via `CFG` (`[render.fire_vfx]`); tuning via
lighting_demo/fire_studio harnesses.

**New systems it creates** (draft rules, to land in CLAUDE.md at
implementation): **FireVFX pass** — `renderer/fire_vfx.py` + shaders: "the
only flame-structure look; reads dequantized fields, writes no synced state,
never runs headless." **Particle primitive** (F4) — "the one particle system
(embers/sparks/debris); future consumers (#34 weapon anims, #40 scorch/blood)
extend it, never build a second."
