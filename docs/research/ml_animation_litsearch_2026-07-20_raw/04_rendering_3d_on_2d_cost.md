# The 3D-model-over-2D rendering pipeline and its real cost (Agent 4)

**Scope:** Can Breach's cheap pyray/raylib top-down 2D game render rigged 3D marines
on top of the world, and what does it cost? Covers raylib's actual glTF/skeletal-animation
capabilities, where to get free rigged humanoids, the runtime cost of skinning and the
crowd-rendering toolbox, the pre-render-to-sprites middle ground, a concrete compute
budget for "tens of marines at 90° top-down / 60 fps / cheap GPU," and a recommended
minimal first pipeline.

Date: 2026-07-20. Capture, not canon.

---

## 1. raylib / pyray 3D + glTF reality check

**Yes, raylib loads rigged glTF natively and animates it.** The core API is exactly the
one Erik named:

- `LoadModel(path)` — loads a `.glb`/`.gltf` (also `.m3d`, `.iqm`, `.vox`, `.obj`) into a `Model`.
- `LoadModelAnimations(path, &count)` — returns an array of `ModelAnimation` (the clips baked in the glTF).
- `UpdateModelAnimation(model, anim, frame)` — advances the pose for a given clip + frame (CPU skinning path).
- `DrawModel` / `DrawModelEx` — draw with position, rotation axis+angle, scale, tint.

glTF/GLB is the **recommended** rigged format for raylib (round-trips cleanly through
Blender). Caveat worth flagging: raylib's glTF *animation* importer has a long tail of
edge-case bugs — several open issues on scale-animated bones, multi-mesh glTF, and
incorrect transforms (issues #4055, #4569, #4888, #1623). Practically: **test each asset**;
a clean single-skeleton humanoid from Mixamo/Quaternius almost always works, exotic rigs
sometimes don't. [flag: quality varies by asset]

### CPU vs GPU skinning — the single most important fact

**Until raylib 5.5 (Nov 2024), raylib only did CPU skinning**, and it was the bottleneck.
The maintainers' own issue #4587 states plainly: *"It is not possible to run more than
~20 animated characters"* with CPU skinning. CPU skinning transforms every vertex on the
CPU each frame, and in **Python (pyray) the per-call C-marshalling overhead makes this
worse** — the pyray docs explicitly warn every call into C is costly and recommend PyPy or
moving inner loops to the raw `raylib.*` functions.

**raylib 5.5 added optional GPU skinning** (PR #4321 by Daniel Holden / `orangeduck`, the
person who has pushed raylib animation furthest). What it does:

- For animated meshes, `boneIds` and `boneWeights` vertex attributes are now uploaded to the GPU.
- Each mesh gets a `boneMatrices` array uploaded as a **shader uniform** (`uniform mat4 boneMatrices[MAX_BONE_NUM];`).
- **`UpdateModelAnimationBoneMatrices(model, anim, frame)`** replaces `UpdateModelAnimation` — instead of transforming vertices on the CPU, it just computes the per-bone matrices and hands them to the shader; the **skinning happens in the vertex shader**.
- You supply a small skinning vertex shader; the stock `models_gpu_skinning` example ships one that works on desktop GL and GLES2 (Raspberry Pi). The example is also available in `pyray` form.
- CPU work per character drops to "compute N bone matrices + upload a uniform"; the per-vertex transform (the expensive part) moves to the GPU where it costs **<1 ms** even for many characters.
- 5.5 *also* sped up the remaining CPU path by simplifying the math.

**Two caveats on the GPU path:**
1. `MAX_BONE_NUM` is a fixed uniform-array size (limited-memory platforms want it small). A humanoid needs ~30–60 bones, comfortably under typical 128 limits.
2. The current raylib API assumes **one animation instance per `Model`**. For many *identically-posed* units that's fine; for many units in *different* poses you either keep a `boneMatrices` set per unit (cheap — it's just matrices) or duplicate the lightweight `Model` handle. The maintainers acknowledge this API is "weak" for GPU crowds (discussion #4606, ongoing redesign). [flag: API in flux]

**Bottom line:** on raylib ≥5.5 with the GPU-skinning path, "tens of animated marines"
is a non-issue. If you were stuck on CPU skinning (older raylib, or you never switch to
`UpdateModelAnimationBoneMatrices`), ~20 characters is the practical ceiling — and lower
in CPython.

### Compositing 3D into the 2D top-down world

raylib mixes 3D into a 2D scene cleanly, and it fits Breach's world-space render-target model:

```
BeginTextureMode(worldRT)        # Breach's existing world-space target
    # ... draw 2D ground/tiles/props in world coords ...
    BeginMode3D(topDownCamera)   # camera positioned above, looking straight down (−Y)
        # draw each marine: DrawModelEx(model, worldPos3D, upAxis, facingDeg, scale, tint)
    EndMode3D()
    # ... draw 2D overlays (selection rings, HP bars) ...
EndTextureMode()
# then the usual single camera blit of worldRT to screen
```

Key points:
- `BeginMode3D`/`EndMode3D` can be nested **inside** `BeginTextureMode`, so 3D marines
  composite into the same world-space RT as the 2D layers — Breach's "camera = one
  viewport blit" model is preserved.
- A `RenderTexture2D` **has a depth buffer**, so the 3D marines depth-sort among
  themselves correctly. For a true 90° top-down look, use an **orthographic** camera
  (`camera.projection = CAMERA_ORTHOGRAPHIC`) placed directly above the scene.
- **Sorting/occlusion vs the 2D world:** the 2D layers have no depth, so ordering between
  2D props and 3D units is by draw order (painter's algorithm) — draw ground first, then
  units. Continuous `facing` (float radians) maps directly to the model's yaw angle — one
  of the big wins of live 3D over sprites (no 8-direction snapping). For unit-vs-unit
  overlap, either rely on the 3D depth buffer or sort by world-Y before drawing.
- Advanced needs (writing custom depth, hybrid 2D-depth + 3D) are covered by the
  `shaders_hybrid_rendering` and `shaders_depth_writing` examples, but Breach almost
  certainly won't need them for flat top-down.

---

## 2. Getting the assets (free rigged humanoids)

You do not need to model or rig anything. Three strong CC0/royalty-free sources:

**Mixamo (Adobe)** — the pragmatic winner for "marines first":
- Free, no subscription. **Auto-rigs** an uploaded humanoid mesh (place ~6 markers) and
  gives a large library of mocap animations (idle, walk, run, aim, fire, hit, death…).
- **Licensing:** royalty-free for personal/commercial/non-profit, including video games;
  no credit required. The one restriction that matters: you **can't redistribute the raw
  character/animation files as standalone assets** — they must be baked into your project
  (a game build is fine). Export as **glTF/GLB or FBX**.
- **One caveat directly relevant to Breach:** Mixamo's terms prohibit using its content to
  **train machine-learning models**. Breach is render-only for the marines — the models
  never enter the headless ML training loop — so this is fine, *but keep the 3D-render
  layer strictly separate from any training data pipeline* and don't feed rendered frames
  of Mixamo characters into model training. [flag: ML-training clause — real, but avoidable by keeping render-only]

**Quaternius** — best CC0 option, zero licensing friction:
- Thousands of low-poly, game-ready, **rigged + animated** models under **CC0** (public
  domain — commercial OK, no attribution, ML-training OK). Available in **glTF**, FBX, OBJ.
- The **Universal Animation Library** is especially useful: a shared humanoid skeleton with
  a big locomotion/action set (8-direction walk, jog, sprint, crawl, swim, sit, death…),
  so you can **retarget the same animation set onto multiple downloaded meshes**.

**Kenney** — CC0, stylized:
- "Animated Characters" packs, CC0, rigged/animated, glTF/FBX. Same liberal terms as Quaternius.

**Sketchfab** — filter to CC0 / CC-BY; huge variety, but per-asset rig quality varies and
CC-BY requires attribution. Good for the "later, larger monsters" phase.

**Retargeting note:** if you standardize on one skeleton (Mixamo's, or Quaternius'
Universal), you can share one animation set across many meshes — marine, then heavy, then
monster — instead of sourcing animations per model. This is the cheapest content pipeline
long-term.

---

## 3. Runtime cost of skeletal animation + the crowd toolbox

**Per-character GPU-skinning cost (the modern path):**
- CPU: compute ~30–60 bone matrices for the current pose + upload one uniform block → microseconds.
- GPU: one draw call, vertex shader does `pos = Σ weight_i · boneMatrix[i] · pos`
  (typically 4 weights/vertex). For low-poly humanoids (~1–5k tris) this is trivially cheap.
- Memory: a small VRAM cost for the `boneIds`/`boneWeights` attributes (paid once at load).

**Where cost actually comes from, in order of likely pain for Breach:**
1. **CPU skinning** (if you don't switch to the GPU path) — the ~20-character wall. *Mitigation: use `UpdateModelAnimationBoneMatrices` + skinning shader.*
2. **Draw calls** — one per model per frame. Tens of marines = tens of draw calls = nothing on any GPU. This only becomes a problem in the thousands. *Mitigation: instancing (see below), and only if you scale way up.*
3. **Shadows** — real-time shadow maps mean an extra render pass per light and roughly double the draw calls. This is the classic hidden cost. *Mitigation: use a cheap **blob shadow** (a soft dark sprite/decal under each unit). Reads perfectly in top-down and costs one textured quad.*
4. **Bone count** — higher bone counts inflate the uniform upload and shader work slightly, and can hit `MAX_BONE_NUM`. *Mitigation: keep humanoids at 20–40 bones.*
5. **Overdraw / fill** — negligible for small on-screen characters at top-down scale.

**The crowd-rendering toolbox** (relevant only when Breach scales to hundreds+; overkill
for tens, but Erik asked and wants bigger battles later):

- **GPU instancing of skinned meshes** — draw many identical meshes in one call. raylib has
  `DrawMeshInstanced` for static meshes; skinned instancing needs the animation-texture
  trick below (raylib's stock GPU-skinning uniform path is per-model, not per-instance).
- **Vertex/Animation Texture (VAT / animation baking)** — bake every frame of an animation's
  bone transforms (or final vertex positions) into a **texture**; the vertex shader reads
  the pose from the texture indexed by time + instance. Combined with GPU instancing this
  renders **~10,000 characters in ~20 draw calls**. Reference implementation:
  chenjd's *Render-Crowd-Of-Animated-Characters* (Unity, but the technique is engine-agnostic
  and portable to raylib via a custom shader). **Trade-off:** animation **blending is hard**
  with baked textures (you're sampling discrete baked poses), so it suits background crowds,
  not precise hero animation.
- **NVIDIA GPU Gems 3, Ch. 2 "Animated Crowd Rendering"** — the canonical reference:
  instanced palette skinning + vertex-texture fetch rendered **~10,000 independently
  animating characters at 30 fps on a GeForce 8800 GTX** — that's **2007** hardware. A
  modern "cheap" GPU eats this. This is the strongest evidence that Breach's scale is not
  remotely a problem.
- **Impostors / billboards / LOD** — at distance, replace the skinned mesh with a
  pre-rendered billboard (impostor) that swaps by view angle. "Impostors and
  pseudo-instancing for GPU crowd rendering" (Millán/Rudomín) and AC-style hybrid systems
  use a LOD map to pick geometry-near vs impostor-far. For a top-down game where all units
  are at similar (small) screen size, a **single global LOD** decision (or none) is enough.
- **AI LOD (Assassin's Creed Unity, GDC 2015)** — 10,000 persistent NPCs via aggressive
  AI+animation LOD and pooling. Not needed at Breach's scale, but the "recycle/pool a fixed
  set of animated instances" idea is a good pattern if unit counts ever explode.

**How many can a modest GPU push at 60 fps?** Ballpark, with GPU skinning:
- Naïve one-draw-call-per-model, low-poly: **hundreds** of skinned characters at 60 fps on a modest discrete/integrated GPU, draw-call-bound long before GPU-bound.
- Instanced + animation-texture: **thousands to ~10,000** (per the 2007 GPU Gems number, higher today), at the cost of blending flexibility.
- CPU skinning (raylib pre-5.5 path): **~20** — and fewer in CPython.

Tens of marines sits so far below every one of these ceilings that the honest answer to
Erik's central worry is: **no, it is not expensive, provided you use the GPU-skinning path.**

---

## 4. The pragmatic middle ground: pre-render 3D → 2D sprite sheets

The classic Diablo / Diablo II / StarCraft approach: build/animate in 3D, then **render
the model offline** from a fixed camera into **sprite sheets** (N directions × M
animation frames), and ship only the sprites. The runtime then stays exactly Breach's
current cheap 2D path — static PNGs, zero runtime 3D, zero skinning.

**How to pre-render (offline, in Blender or a tiny raylib script):**
- Orthographic camera at Breach's exact top-down angle (90°, straight down — or a slight
  tilt if you want to see the models' fronts).
- For each animation clip, step the frames and **snapshot** each; repeat for each facing
  direction (rotate model or camera). 8 or 16 directions is standard for
  iso/top-down.
- Assemble into sprite sheets, drop into the existing 32×32 (or larger) sprite pipeline.

**Honest comparison vs live 3D:**

| | Live GPU-skinned 3D | Pre-rendered sprites |
|---|---|---|
| Runtime cost | low (tens of units trivial) | ~zero (current path) |
| Facing | **continuous** (matches Breach's float radians) | snapped to 8/16 dirs |
| Per-unit variety (tint/pose/damage) | easy (live) | baked; needs re-render or palette tricks |
| Lighting | dynamic possible | baked at render time |
| Memory | one mesh + anims | dirs × frames × units of PNGs (can balloon) |
| Iteration | change model, done | re-run the pre-render bake |
| Big monsters later | scale mesh, reuse pipeline | huge sprites × many frames = big atlases |
| Risk | glTF import bugs, shader work | none (it's just sprites) |

**Hybrids worth knowing:**
- **(b) Live 3D for hero/nearby units, sprites for the crowd** — draw the few important
  marines as skinned 3D, distant/filler units as impostor sprites.
- **(c) 8-direction *live* 3D** — snap the model's yaw to 8 directions instead of continuous;
  pointless for Breach since continuous 3D is free and Breach already has continuous
  `facing`.

For Breach specifically, the **continuous `facing` float** is a strong argument *for* live
3D: pre-rendered sprites throw that away (back to 8/16-direction snapping — the very
limitation sprites impose), while live 3D consumes it directly as a yaw angle.

---

## 5. Compute-cost reality check — "tens of animated 3D marines, 90° top-down, 60 fps, cheap GPU"

**Verdict: comfortably affordable on the GPU-skinning path. This is not an expensive ask.**

Concrete per-frame budget for, say, 40 marines (raylib ≥5.5, GPU skinning, low-poly ~2k-tri meshes, ~35 bones):
- **CPU:** 40 × (advance animation frame + compute ~35 bone matrices + upload uniform) — sub-millisecond even in CPython; the matrix math is tiny and the only C-boundary crossings are the per-model update + draw calls (~80 calls total). Fine for pyray.
- **GPU:** 40 draw calls, each a small skinned mesh in the vertex shader — **well under 1 ms** of GPU skinning; fill/raster negligible at top-down scale.
- **Frame headroom:** Breach targets 60 fps (16.7 ms). Skinned marines consume a low-single-digit-millisecond fraction of that at most. The existing 2D compositing dominates the frame far more than the marines do.

**Where it would get expensive (and the cheapest fix for each):**
- **Staying on CPU skinning** → ~20-unit wall, worse in Python. *Fix: `UpdateModelAnimationBoneMatrices` + the stock skinning shader.* (biggest lever)
- **Real-time shadow maps** → extra pass, ~2× draw calls. *Fix: blob-shadow sprite per unit.*
- **Per-unit unique `Model` copies** (memory + load time) → *Fix: share one loaded `Model`, vary only transform/tint/bone-matrices per unit.*
- **Scaling to hundreds+ later** → draw-call bound. *Fix: instancing + animation-texture (Section 3); not needed for tens.*
- **pyray per-call overhead in a hot loop** → *Fix: batch, use raw `raylib.*` in the inner loop, or PyPy — but at tens of units this won't bite.*

The dominant risk for Breach is **not** performance — it's **asset/import friction** (glTF
animation quirks) and the render-layer integration work, both one-time.

---

## 6. Recommended minimal pipeline — "marines first, learn a little"

A concrete first step that stays true to Breach's cheap, render-only, top-down design:

1. **Assets:** grab one low-poly rigged humanoid + a small clip set (idle, walk, fire,
   death). Fastest path: **Quaternius** (CC0, no strings, glTF, Universal Animation Library)
   or **Mixamo** (auto-rig + rich mocap, export GLB) — Mixamo if you want polished mocap,
   Quaternius if you want zero licensing thought. Keep bones ≤40, tris ≤~3k.
2. **Engine:** confirm raylib **≥5.5** under pyray. Load with `LoadModel` +
   `LoadModelAnimations`. **Use the GPU-skinning path** from the start:
   `UpdateModelAnimationBoneMatrices` + the `models_gpu_skinning` example's skinning
   vertex shader. (Don't build on CPU `UpdateModelAnimation` — you'd hit the 20-unit wall.)
3. **State→animation mapping:** pick the clip from unit state (idle / moving / firing /
   dying) and advance its frame each render tick. This is **render-only, driven off the
   sim state read-only** — no determinism impact, and it stays gated OFF in headless ML
   training.
4. **Placement + facing:** set each marine's transform from its world position; map the
   continuous `facing` radians straight to the model's **yaw**. 48 world-px/tile,
   ~3×3-tile footprint → pick a model scale so the mesh matches the sprite footprint.
5. **Compositing:** inside Breach's existing world-space RT, after the 2D ground/props,
   open `BeginMode3D` with an **orthographic top-down camera** and draw the marines, then
   `EndMode3D` and continue with 2D overlays. The single camera blit is unchanged.
6. **Shadows:** skip shadow maps; draw a **blob-shadow sprite** under each marine.
7. **Fallback ready:** if glTF import fights you or you want zero runtime cost, the
   **pre-render-to-sprite** bake (Section 4) reuses the *same* Mixamo/Quaternius asset —
   render 8–16 directions × frames offline into the existing sprite path. Keep this in your
   back pocket; it de-risks the whole experiment.

**Recommendation:** start with **live GPU-skinned 3D** (step 1–6). It directly uses
Breach's continuous `facing`, scales to tens of marines with huge headroom, reuses one
mesh + shared animations for the later "bigger monsters," and the only real cost is a
one-time integration + asset-vetting pass — not per-frame compute. Treat pre-rendered
sprites as the safety net, not the default.

---

## 7. If you read three things

1. **raylib `models_gpu_skinning` example + PR #4321** (Daniel Holden / orangeduck) —
   the exact API (`UpdateModelAnimationBoneMatrices` + skinning shader) Breach should build
   on, and the thing that turns "~20 characters" into "hundreds."
   <https://github.com/raysan5/raylib/pull/4321> · <https://www.raylib.com/examples/models/loader.html?name=models_gpu_skinning>
2. **NVIDIA GPU Gems 3, Ch. 2 "Animated Crowd Rendering"** — 10,000 animated characters at
   30 fps on 2007 hardware; the definitive proof that Breach's scale is trivially cheap, plus
   the instancing + vertex-texture technique for the "big battles later" phase.
   <https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-2-animated-crowd-rendering>
3. **Quaternius Universal Animation Library** (CC0 assets) and the **Mixamo FAQ** (licensing)
   — the two content pipelines, one friction-free CC0, one polished-mocap-but-no-ML-training.
   <https://quaternius.com/packs/universalanimationlibrary.html> · <https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html>

---

## 8. Flagged uncertainties & citations

**Flags:**
- raylib's **glTF animation importer has known bugs** on non-trivial rigs (scale-animated
  bones, multi-mesh). Vet each asset; a clean single-skeleton humanoid is the safe case.
  (Issues #4055, #4569, #4888, #1623, #4412.)
- raylib's GPU-skinning API is **one-animation-instance-per-`Model`** and acknowledged
  "weak" for large GPU crowds; a redesign is in discussion (#4606). Fine for tens of units;
  revisit before doing thousands.
- **Mixamo forbids using its content to train ML models.** Breach is render-only for the
  marines, so this is avoidable — but keep the 3D-render layer strictly out of any training
  data path. Quaternius/Kenney (CC0) have no such restriction.
- The "hundreds of skinned characters on a modest GPU at 60 fps" figure is an
  **extrapolation** from the GPU Gems 3 (2007) result and general draw-call/skinning cost —
  not a raylib-specific benchmark. The raylib-specific hard number is the **~20-character
  CPU-skinning ceiling** (maintainers, issue #4587). Tens of marines is safe under either
  reading.

**Sources:**
- raylib animation performance / CPU ceiling — issue #4587: <https://github.com/raysan5/raylib/issues/4587>
- Optional GPU skinning — PR #4321: <https://github.com/raysan5/raylib/pull/4321>
- GPU skinning example: <https://www.raylib.com/examples/models/loader.html?name=models_gpu_skinning>
- Animation-system redesign discussion — #4606: <https://github.com/raysan5/raylib/discussions/4606>
- Animation blending PR #4578: <https://github.com/raysan5/raylib/pull/4578>
- glTF import/animation bugs: #4055, #4569, #4888, #1623, #4412 (raysan5/raylib issues)
- raylib 5.5 release notes: <https://github.com/raysan5/raylib/releases/tag/5.5>
- pyray bindings + examples: <https://pypi.org/project/raylib/> · <https://github.com/blep/pyray_examples>
- Compositing 3D in render texture / hybrid rendering: <https://www.raylib.com/examples/shaders/loader.html?name=shaders_hybrid_rendering> · discussion #2373 (layered 3D) · #1579 (RenderTexture depth)
- Mixamo licensing FAQ: <https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html> · community FAQ: <https://community.adobe.com/questions-696/mixamo-faq-licensing-royalties-ownership-eula-and-tos-589400>
- Quaternius (CC0, glTF, Universal Animation Library): <https://quaternius.com/> · <https://quaternius.com/packs/universalanimationlibrary.html>
- Kenney Animated Characters (CC0): <https://kenney-assets.itch.io/animated-characters-3>
- GPU Gems 3, Ch.2 Animated Crowd Rendering (~10k @30fps, 8800 GTX): <https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-2-animated-crowd-rendering>
- Animation-texture crowd instancing (chenjd, ~10k in ~20 draw calls): <https://github.com/chenjd/Render-Crowd-Of-Animated-Characters>
- GPU skinning + instancing 1000+ chars (BigBro222): <https://big-bro222.github.io/blog/gpu-skinning/>
- Impostors & pseudo-instancing for GPU crowds: <https://dl.acm.org/doi/10.1145/1174429.1174436>
- Assassin's Creed Unity massive crowd / AI LOD (GDC): <https://gdcvault.com/play/1022411/Massive-Crowd-on-Assassin-s>
- Real-Time Large Crowd Rendering on GPU (Dong 2019): <https://onlinelibrary.wiley.com/doi/10.1155/2019/1792304>
- Pre-render 3D→2D sprite technique (Diablo/StarCraft), ortho camera + 8/16 dirs: gamedev.net topics 666043, 663241
