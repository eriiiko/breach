# Marine Shader Foundation — design doc

**Status: DESIGN, awaiting Erik's approval (2026-07-20).** Branch `anim-phase0-3d-marines`.
Feel-adjacent → **HUMAN-TEST gate, no auto-merge.** Evidence base: the focused lit search in
[`research/marine_shader_foundation_2026-07-20_raw/`](research/marine_shader_foundation_2026-07-20_raw/)
(01 skinning mechanics · 02 lighting+normal-maps · 03 Breach lighting integration) plus a direct
introspection of the Quaternius asset. Builds on Phase 0 ([`anim_phase0_impl_2026-07-20.md`](anim_phase0_impl_2026-07-20.md)).

## Goal
Make the 3D marines a **strong, lit render foundation**: lit by Breach's raycast lights so they sit in
the scene (not flat-tinted), with a clean seam for later normal maps and GPU skinning. This is the
"marines' normal surface interacts with our raycasted lights + textures (maybe normal maps)" item.

## v2 — hardened plan after adversarial critique (2026-07-20)

Three independent critics (shader-math, integration, scope) stress-tested v1. **No blockers to the
approach**; two plumbing blockers, several must-fix-before-feel bugs, and a cheaper sequencing. This
section supersedes the patch plan below.

### Confirmed good (survived critique)
- Deferring GPU skinning — empirically confirmed CPU skinning is live (`animVertices/animNormals`
  allocated, no bone-weight VBO) and a custom shader lights posed vertices cleanly.
- Reusing Breach's baked light textures — `LightingPass.light_tex_a/b` are public and uploaded
  (`upload_state`) *before* units draw (`compose_world`). No raycaster reinvention.
- The 2D→3D light-dir remap `L = vec3(dir.x, u_light_z, dir.y)` is **correct** (math critic couldn't break it).
- Tangent-free derivative TBN for P2; no determinism/sim/regression risk.

### Blockers (resolved on paper)
- **B1 — texture binding.** Light textures CANNOT be pre-bound as uniforms before `draw_model_ex`
  (`DrawMesh` manages its own texture units and clobbers them). **Fix:** put `light_tex_a/b` in
  **material map slots** (METALNESS=1, NORMAL=2; albedo stays ALBEDO=0), samplers `texture1`/`texture2`,
  auto-bound every draw. Scalars/matrices via `set_shader_value` once/frame are fine.
- **B2 — materials + matrices.** Meshes use materials **1 and 2** (material 0 is dead) — set the custom
  shader on both. The shader must declare `matModel`/`matNormal` by name (stock model shader has them at
  loc −1) for the world-space normal, and must **NOT** declare `boneMatrices` (would double-transform the
  CPU-skinned mesh). *The spike verifies matModel/matNormal actually upload in this wheel; fallback = set
  the model matrix as a per-frame uniform.*

### Bugs (fold into P1)
- **Pre-normalized light direction** → the marine's *side* normals get an arbitrary lit direction under
  near-overhead lamps (the flat ship floor was immune). **Resolution:** adopt **half-Lambert**
  (`N·L*0.5+0.5`) — which the scope critic independently wanted for top-down legibility — it softens the
  instability AND reads better. The "proper" fix (plumb a directionality magnitude through `lighting.py`)
  touches the golden-gated ship system → **deferred unless half-Lambert proves insufficient. [Erik's call.]**
- **Sample the light at the marine's ground position** (per-fragment world-XZ, ignore Y — no per-unit
  uniform needed). Log the "animated limb crossing a light boundary may shimmer" risk; foot-tile-uniform
  is the fallback knob.
- **world_uv Y-orientation:** ship samples over a y-up RT quad; marine `worldPos.z` is y-down → a likely
  v-flip to reconcile. De-risk in the spike.
- **Don't over-claim parity:** same colour/occlusion/tone-map, but genuinely *different* directional
  shading (the ship is directionless-flat with no normal map). **Don't copy the vacuum `discard`**;
  **sRGB-decode the albedo** (`u_srgb_decode=1`) or it double-darkens; fragment must output **alpha=1.0**
  (premultiply blit); **drop the flat `light_fn` multiply** in `_draw_one` (the field sample replaces it).
- **GLSL drift:** factor the shared srgb/aces/unpack helpers (string-concat a common snippet), don't copy
  the golden-gated ship GLSL verbatim.

### Legibility (top-down) — into P1, not deferred to P3
Overhead N·L barely varies on a small token. Put the readability wins IN P1: **half-Lambert** + a cheap
**rim** `pow(1-N·V, k)` for silhouette + an **angled/grazing key** (the marine gets its OWN `u_light_z`,
lower than the ship's). These likely matter more than normal maps top-down — so **P2 (normal maps)
becomes genuinely optional, pending the P1 feel-check.**

### Revised sequence (supersedes "The design that falls out" below)
- **Patch 0 — RGB per-unit tint, NO shader (~10 lines).** Sample `light_rgb[cy,cx]` (already CPU-side),
  tint `base_tint × incoming_rgb`, drop the max-collapsed scalar. Closes colour+occlusion+brightness
  parity immediately, improves the shader's own fallback path, is the cheapest feel-probe. Ship + eyeball first.
- **Spike — one marine on one lit tile.** Confirm (a) matModel/matNormal upload, (b) world_uv hits the
  right tile with correct Y-flip (marine darkens in an unlit room, reddens under a red lamp beside a
  matching ship tile), (c) N·L responds to the baked 2D direction, (d) material-slot texture binding works.
- **P1 — lit marine shader** with all fixes (material-slot textures, matModel/matNormal, per-fragment
  world-XZ, sRGB albedo, alpha=1, half-Lambert + rim + angled key, shared GLSL). Feel-gate.
- **P2 — normal maps (OPTIONAL, pending P1 feel).** Derivative TBN; risk: degenerates on the marine's
  vertical *sides* (the silhouette); needs its own y-sign + a sourced/generated normal map.
- **Deferred — GPU skinning** (verify real on-screen count first; ~20-unit CPU ceiling straddles "tens").

### The one decision for Erik
How far to chase *directional* correctness in P1: **(recommended) half-Lambert soften now** — cheap,
fixes legibility, masks the pre-normalized-dir instability, and leaves the ship's golden-gated lighting
untouched — versus investing up front in plumbing a directionality scalar through `lighting.py`. I'd do
half-Lambert now and only plumb directionality if the feel-check demands it.

---

## What the search settled (convergent findings)

1. **GPU skinning is NOT the blocker it looked like — and it's not needed for the look.** The missing
   `UpdateModelAnimationBoneMatrices` is a red herring; the real gate is the compile flag
   `SUPPORT_GPU_SKINNING`, which the shipped raylib-python-cffi wheel has **OFF** (verified: the
   model's bone-weight VBOs are unset). So GPU skinning needs a **custom binding rebuild** — but it
   only buys **unit-count headroom** (lifts the ~20-marine CPU-skinning ceiling), which we don't need
   yet. **Lighting and skinning are independent axes.**

2. **We can ship a lit + (later) normal-mapped marine shader NOW, on CPU skinning, no rebuild.** CPU
   skinning writes correctly-posed **positions and normals** into the mesh VBOs; a custom material
   shader lights those posed vertices with zero bone knowledge. N·L lighting is therefore correct on
   animated marines today.

3. **Breach already bakes lighting into two textures the ship shader samples — reuse them.** The C++
   raycaster marches every light source (occlusion, colour, 1/r falloff) into two per-frame RGBA16F
   textures: `light_tex_a` = incoming RGB + dir.x, `light_tex_b` = smoke-glow + dir.y. `shaders/lighting.fs`
   samples them at `world_uv`, builds `L = normalize(vec3(light_dir_2d, u_light_z))`, does N·L against a
   normal map, then `lit = diffuse*(u_ambient + incoming_rgb*u_light_gain*ndotl)` + ACES + sRGB. The
   marine shader should sample the **same two textures** → marines inherit colour, occlusion, falloff,
   and tone-map, and match the ship exactly. (Rejected: feeding raw `LightSource` uniforms — it
   discards occlusion and duplicates the raycaster.)

4. **Normal mapping without vertex tangents (the asset has none, and CPU skinning wouldn't skin them):**
   reconstruct the TBN in the fragment shader from **screen-space derivatives** of world-pos + UV
   (Schüler's tangent-free normal mapping). This sidesteps the un-skinned-tangent problem entirely and
   needs no tangent data. Reuse Breach's existing normal-map **Y-sign** convention (`u_normal_y_sign`).

5. **Asset reality:** the Quaternius model ships **albedo only, no normal map** (2 meshes, 53 bones,
   shared albedo `tex=1`). So *lit albedo* needs no new art; *normal mapping* needs a sourced/generated
   normal map.

## The design that falls out

**A staged arc on CPU skinning, reusing Breach's baked light field. No toolchain change, no new art for
the first (high-value) cut.**

### Patch 1 — Lit marine shader (the core win; no new asset)
- A custom material shader assigned to the marine model's material (`model.materials[i].shader`), owned
  by `UnitModelRenderer`.
- **Vertex:** trivial — the mesh is already CPU-posed; pass world position, world normal
  (`matNormal = transpose(inverse(model))`), and UV. Compute `world_uv` for the light-texture lookup
  from the unit's world-px position (the same mapping the ship uses).
- **Fragment:** sample `light_tex_a/b` at `world_uv` → reconstruct `incoming_rgb` + 2D light dir → lift
  to 3D `L = normalize(vec3(dir.x, u_light_z, dir.y))` (Y-up remap) → `ndotl = max(0, N·L)` on the
  glTF normal → `lit = albedo*(u_ambient + incoming_rgb*u_light_gain*ndotl)` → ACES + sRGB (reuse the
  ship's GLSL verbatim). Reuse uniforms `u_ambient`, `u_light_gain`, `u_light_z`.
- **Integration:** `UnitModelRenderer` needs a handle to the two light textures (from `lighting.py`),
  bound as uniforms before `draw_units`. Keep the current `light_fn` scalar path as a fallback for when
  the textures aren't available. This lives behind the same `_draw_one` seam.
- **Result:** marines lit, coloured, occluded, and tone-mapped exactly like the ship. This alone
  delivers the "lit surface interacting with the raycast lights" ask.

### Patch 2 — Normal mapping (surface detail; needs a normal-map asset)
- Add screen-space-derivative TBN in the fragment shader; sample a normal map bound to the material's
  NORMAL slot; perturb N before the N·L. Honour `u_normal_y_sign`.
- **Asset:** source or generate a normal map for the marine (the model has none). Options: generate from
  the albedo/height (cheap, approximate) or find a matching CC0 one. Decision needed (below).

### Patch 3 — Top-down readability tuning (feel)
- A straight-overhead light flattens a top-down character (all top faces `N·L≈1`). Add a subtle
  **angled key/fill** for the marines (or lean on scene side-lights + normal detail) so form reads.
  Tunable constants, feel-gated.

### Deferred (not this arc) — GPU skinning
- Purely the **performance lever** (lift the ~20-unit ceiling). Reached later via a
  `-DSUPPORT_GPU_SKINNING=ON` custom binding build, behind the `_draw_one` seam. **The fragment/lighting
  half of the shader is reused unchanged** across CPU and GPU skinning — only the vertex preamble gains
  the bone-skinning of position/normal — so none of Patches 1–3 is wasted when we later add it.

## Why this is the right call now
- Delivers Erik's actual want (lit marines matching the scene) **now**, no rebuild, no new art for P1.
- **Forward-compatible:** the lighting shader is identical whether skinning is CPU or GPU; GPU skinning
  is a clean, deferred perf swap, not a redo.
- Keeps the render/sim/ML separation intact (all render-layer; nothing touches `Unit`/sim/determinism).

## Effort / risk (honest)
- **P1 lit shader:** ~moderate. Main risk is getting the `world_uv` mapping and the 2D→3D light-dir
  remap right so marines match the ship; de-risked by a spike that renders one lit marine next to a lit
  ship tile and compares. Low blast radius (render-only, toggle-gated).
- **P2 normal map:** moderate; risk is the tangent-free TBN + Y-sign + gamma correctness (classic
  failure modes catalogued in digest 02) and sourcing a decent normal map.
- **P3 tuning:** low, pure feel.
- **No determinism/sim risk anywhere.**

## Open questions for Erik (his calls)
1. **Scope of the first cut:** ship **P1 (lit albedo) alone first** and feel-check it before P2 normal
   maps? (Recommended — P1 is the big win and needs no new art; P2 adds polish.) Or go straight to
   lit + normal-mapped?
2. **Normal-map asset (only if P2 now):** generate an approximate one from the albedo, or hold P2 until
   we have a proper normal map? (Recommended: P1 first, decide P2's asset after seeing P1.)
3. **Bank Phase 0 to main first?** Phase 0 (unlit 3D marines, toggle-off default) is a working baseline;
   merging it first gives this arc a clean base. (Recommended.)
4. **Adversarial critique pass** on this doc before building (per our workflow for foundational work)?
   Optional for a render shader; offered.
