# GPU Skinning in raylib + doing it from a cffi binding without the helper

Research digest — Breach marine renderer. 2026-07-20.

Scope: how raylib 5.5/6.x implements GPU skinning end-to-end; whether Breach's
raylib-python-cffi binding (6.1-dev, RAYLIB STATIC 6.0.1.0) can do it despite
lacking `UpdateModelAnimationBoneMatrices`; the independence of lighting from
skinning; and a recommendation.

Primary sources (all read directly, not summarized):
- PR #4321 "[rmodels] Optional GPU skinning" by Daniel Holden (orangeduck):
  https://github.com/raysan5/raylib/pull/4321
- Stock example `examples/models/models_animation_gpu_skinning.c` and shader
  `examples/models/resources/shaders/glsl330/skinning.vs` (raysan5/raylib master).
- `src/rmodels.c` (`UpdateModelAnimation`, `UpdateModelAnimationVertexBuffers`,
  `DrawMesh`/`DrawModelEx`, `UploadMesh`), `src/raylib.h`, `src/rlgl.h`,
  `src/config.h` (raysan5/raylib master, fetched 2026-07-20).
- raylib-python-cffi build workflow `.github/workflows/build.yml`
  (electronstudio/raylib-python-cffi master).
- Redesign context: raylib Discussion #4606 "REDESIGN: Model Animation System".

## Executive summary (the load-bearing finding)

**The missing `UpdateModelAnimationBoneMatrices` is NOT the real blocker — the
compile flag `SUPPORT_GPU_SKINNING` is.** In current raylib the bone-matrix
math and its upload to the shader are *already done for you* by the stock
`UpdateModelAnimation` + `DrawModel` that your binding exposes. What is gated
behind `SUPPORT_GPU_SKINNING` (OFF by default, and the raylib-python-cffi wheel
does NOT turn it on) is the upload/binding of the per-vertex `boneIndices` /
`boneWeights` **vertex attributes**. Without those attributes on the GPU, a
skinning vertex shader reads all-zero bone data and collapses the mesh. So GPU
skinning is effectively unavailable through the stock draw path of the shipped
wheel — not because a Python function is missing, but because the C library was
built with the feature compiled out. Getting it is a *rebuild-the-binding*
task, not a *call-the-missing-function* task.

Separately and importantly: **lighting and skinning are independent axes.** CPU
skinning writes posed positions+normals into the mesh's own VBOs; any custom
material shader then lights those already-posed vertices with zero bone
knowledge. So Breach can ship a custom lit + normal-mapped marine shader NOW on
top of the existing CPU skinning, with no GPU-skinning work at all. GPU skinning
is purely the performance lever for later.

---

## 1. How raylib does GPU skinning end-to-end

### The data plumbing
A rigged mesh in raylib carries, per vertex, up to 4 bone influences:
- `Mesh.boneIndices` — `unsigned char*`, 4 per vertex (raylib.h calls the
  shader attribute location 6).
- `Mesh.boneWeights` — `float*`, 4 per vertex (location 7).

And a per-frame array of bone transforms:
- the runtime bone matrices, one `Matrix` per bone, uploaded to the shader
  uniform `boneMatrices[MAX_BONE_NUM]` at shader location
  `SHADER_LOC_MATRIX_BONETRANSFORMS` (uniform name
  `RL_DEFAULT_SHADER_UNIFORM_NAME_BONEMATRICES` = `"boneMatrices"`).

> **Struct-location caveat (version-sensitive).** PR #4321 (the design your
> binding's field names match) put the bone matrices on the **mesh**:
> `Mesh.boneMatrices` + `Mesh.boneCount`, uploaded as
> `rlSetUniformMatrices(locs[SHADER_LOC_MATRIX_BONETRANSFORMS], mesh.boneMatrices, mesh.boneCount)`.
> Current raylib master (post "Model Animation System" redesign, Disc. #4606)
> **moved them to the Model**: `Model.boneMatrices` + `Model.skeleton.boneCount`
> + `Model.skeleton.bindPose` + `Model.currentPose`, uploaded as
> `rlSetUniformMatrices(..., model.boneMatrices, model.skeleton.boneCount)`.
> Your binding (6.0.1.0-era, described with `Mesh.boneCount` and the
> `keyframeCount`/`keyframePoses` animation fields) sits between these: it has
> the renamed `SHADER_LOC_MATRIX_BONETRANSFORMS` and mesh-level bone data.
> Whichever it is, the *mechanism* below is identical; only the struct path to
> `boneCount`/`boneMatrices` differs. Confirm the exact layout in a REPL
> (`ffi` struct introspection) before writing code against it.

### The bone-matrix math (raylib computes this for you)
`UpdateModelAnimation(model, anim, frame)` — which your binding DOES expose —
loops every bone and computes, per bone:

```
bindPoseMatrix    = S(bind.scale) * R(bind.rot) * T(bind.trans)   // rest pose
currentPoseMatrix = S(cur.scale)  * R(cur.rot)  * T(cur.trans)    // this frame
boneMatrices[b]   = inverse(bindPoseMatrix) * currentPoseMatrix
```

That `inverse(bind) * current` is exactly the "skinning matrix" a GPU skinning
shader needs — it maps a bind-pose vertex to its posed position. So **the CPU
math to produce the uniform is already run every time you call
`UpdateModelAnimation`**, regardless of CPU vs GPU skinning. (verbatim from
rmodels.c `UpdateModelAnimation`.)

### The upload (also already done for you)
In `DrawMesh` (called by `DrawModel`/`DrawModelEx`), raylib does — and this line
is **NOT** guarded by `SUPPORT_GPU_SKINNING`:

```c
if ((mat.shader.locs != NULL) &&
    (mat.shader.locs[SHADER_LOC_MATRIX_BONETRANSFORMS] != -1) &&
    (model.boneMatrices != NULL))
{
    rlEnableShader(mat.shader.id);
    rlSetUniformMatrices(mat.shader.locs[SHADER_LOC_MATRIX_BONETRANSFORMS],
                         model.boneMatrices, model.skeleton.boneCount);
}
```

So if your material's shader declares the `boneMatrices` uniform, the per-frame
matrices are pushed to it automatically on every draw. No manual uniform upload
needed.

### The skinning vertex shader (stock `glsl330/skinning.vs`, verbatim)
```glsl
#version 330

#define MAX_BONE_NUM 128

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec4 vertexColor;
in vec3 vertexNormal;
in vec4 vertexBoneIndices;   // attribute location 6
in vec4 vertexBoneWeights;   // attribute location 7

uniform mat4 mvp;
uniform mat4 matNormal;
uniform mat4 boneMatrices[MAX_BONE_NUM];

out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;

void main()
{
    int boneIndex0 = int(vertexBoneIndices.x);
    int boneIndex1 = int(vertexBoneIndices.y);
    int boneIndex2 = int(vertexBoneIndices.z);
    int boneIndex3 = int(vertexBoneIndices.w);

    vec4 skinnedPosition =
        vertexBoneWeights.x*(boneMatrices[boneIndex0]*vec4(vertexPosition, 1.0)) +
        vertexBoneWeights.y*(boneMatrices[boneIndex1]*vec4(vertexPosition, 1.0)) +
        vertexBoneWeights.z*(boneMatrices[boneIndex2]*vec4(vertexPosition, 1.0)) +
        vertexBoneWeights.w*(boneMatrices[boneIndex3]*vec4(vertexPosition, 1.0));

    vec4 skinnedNormal =
        vertexBoneWeights.x*(boneMatrices[boneIndex0]*vec4(vertexNormal, 0.0)) +
        vertexBoneWeights.y*(boneMatrices[boneIndex1]*vec4(vertexNormal, 0.0)) +
        vertexBoneWeights.z*(boneMatrices[boneIndex2]*vec4(vertexNormal, 0.0)) +
        vertexBoneWeights.w*(boneMatrices[boneIndex3]*vec4(vertexNormal, 0.0));
    skinnedNormal.w = 0.0;

    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = normalize(vec3(matNormal*skinnedNormal));

    gl_Position = mvp*skinnedPosition;
}
```

This is the standard linear-blend skinning:
`skinnedPos = Σ_i weight[i] * boneMatrices[boneId[i]] * pos`, and the same for
the normal with `w = 0` (direction, not point). Note it uses the *raw*
bind-pose `vertexPosition` — because with GPU skinning the position VBO is never
CPU-transformed. This is the crux of the mutual exclusion (see §3 caveat).

### The compile-time gate (the actual blocker)
In `UploadMesh` and in the `DrawMesh` attribute-binding block, the bone vertex
attributes are wrapped in `#if SUPPORT_GPU_SKINNING`:

```c
#if SUPPORT_GPU_SKINNING
    if (mesh->boneIndices != NULL) {
        mesh->vboId[...BONEINDICES] = rlLoadVertexBuffer(mesh->boneIndices,
            mesh->vertexCount*4*sizeof(unsigned char), dynamic);
        rlSetVertexAttribute(...BONEINDICES, 4, RL_UNSIGNED_BYTE, 0, 0, 0);
        rlEnableVertexAttribute(...BONEINDICES);
    } ...
#endif
```

And `src/config.h`:
```c
#ifndef SUPPORT_GPU_SKINNING
    // GPU skinning disabled by default, some GPUs do not support more than 8 VBOs
    #define SUPPORT_GPU_SKINNING        0
#endif
```

Conversely, the CPU anim buffers (`animVertices`, `animNormals`) are allocated
only under `#if !SUPPORT_GPU_SKINNING`. So the two modes are **mutually
exclusive by compile flag**: with the flag ON, bone attributes go to the GPU and
CPU skinning becomes a no-op; with it OFF (the default), CPU skinning runs and
bone attributes never reach the GPU.

---

## 2. The manual path when the binding lacks `UpdateModelAnimationBoneMatrices`

### Verdict: the missing function is a red herring; the missing *vertex
### attributes* are the problem.

Re-framing the question precisely against what the wheel actually ships:

1. **Do we need to compute bone matrices ourselves?** No. The exposed
   `UpdateModelAnimation` already fills `boneMatrices` (`inv(bind)*current`)
   every frame. **Do we need to upload them?** No. `DrawModel`/`DrawModelEx`
   already push them to `SHADER_LOC_MATRIX_BONETRANSFORMS` (that upload is not
   behind the compile flag). So `set_shader_value_v` / a manual
   `rl_set_uniform_matrices` call is unnecessary for the matrices. This is why
   the absence of `UpdateModelAnimationBoneMatrices` doesn't actually cost you
   anything on the matrix side.

2. **What is genuinely missing?** The wheel is built with
   `SUPPORT_GPU_SKINNING = 0` (the raylib-python-cffi `build.yml` sets
   `-DCUSTOMIZE_BUILD=ON` with several `SUPPORT_*` flags but does **not** pass
   `-DSUPPORT_GPU_SKINNING=ON`, so config.h's default 0 stands). Therefore:
   - `Mesh.boneIndices` / `Mesh.boneWeights` are never uploaded as VBOs, and
   - they are never bound to shader attribute locations 6/7.
   A skinning `.vs` would read the *default* attribute values (0,0,0,0) for both
   indices and weights → `skinnedPosition = 0` for every vertex → the mesh
   collapses to the origin. **GPU skinning through the stock path is not
   possible with this wheel as built.**

### Can we bridge the gap in pure Python (no rebuild)? Feasible but ugly.
The binding does expose the rlgl primitives needed (`rl_load_vertex_buffer`,
`rl_set_vertex_attribute`, `rl_enable_vertex_attribute`,
`rl_enable_vertex_array`, `rl_set_uniform_matrices`), and `Mesh` exposes
`boneIndices`, `boneWeights`, `vaoId`, `vboId`. So in principle:

```
# once, after LoadModel, per mesh:
rl_enable_vertex_array(mesh.vaoId)                      # bind the mesh's VAO
vbo_idx = rl_load_vertex_buffer(mesh.boneIndices, n*4*1, False)   # u8 x4
rl_set_vertex_attribute(6, 4, RL_UNSIGNED_BYTE, False, 0, 0)
rl_enable_vertex_attribute(6)
vbo_w  = rl_load_vertex_buffer(mesh.boneWeights, n*4*4, False)    # f32 x4
rl_set_vertex_attribute(7, 4, RL_FLOAT, False, 0, 0)
rl_enable_vertex_attribute(7)
rl_disable_vertex_array()
# then assign skinning shader to the material; DrawModel uploads boneMatrices.
```

Two hard problems make this fragile:
- **Double transform.** With `SUPPORT_GPU_SKINNING=0`, `UpdateModelAnimation`
  still runs the CPU skinning path (`UpdateModelAnimationVertexBuffers` posed
  positions into the position VBO). If the shader *also* applies `boneMatrices`,
  every vertex is transformed twice. You'd have to stop CPU skinning — but the
  only exposed call that computes the bone matrices is the same call that does
  the CPU skinning. You'd end up re-implementing the `inv(bind)*current` loop in
  Python (from `keyframePoses` + bind pose) purely to avoid the bundled CPU
  step, and uploading the matrices yourself via `rl_set_uniform_matrices` —
  which is exactly the work the compile flag would give you for free and
  correctly.
- **Attribute location assumptions.** Locations 6/7 must match what the shader
  binds; on some drivers you must bind attribute locations before/around link.
  Manual VAO surgery from Python is workable but easy to get subtly wrong, and
  it's per-mesh, per-model bookkeeping.

**Bottom line for §2:** technically possible without a rebuild, but it means
hand-reimplementing the bone-matrix loop *and* hand-managing VBO/VAO state to
dodge a double transform — high friction, fragile across drivers. The clean fix
is a one-line build change: rebuild raylib-python-cffi (or a local raylib) with
`-DSUPPORT_GPU_SKINNING=ON`, after which the stock `UpdateModelAnimation` +
`DrawModel` + `skinning.vs` "just work" and CPU skinning auto-disables. **What's
missing is a compile flag, not Python code.**

---

## 3. THE KEY SEPARABILITY — lighting and skinning are INDEPENDENT axes

This is the most important practical point for Breach, so stating it plainly:

**CPU skinning and a custom lit/normal-mapped shader compose freely. You do NOT
need GPU skinning to get a custom marine shader.**

Why, concretely, in raylib:
- `UpdateModelAnimation` (CPU path) computes posed positions into
  `mesh.animVertices` and posed normals into `mesh.animNormals`, then
  `rlUpdateVertexBuffer`s them into the mesh's **position VBO** and **normal
  VBO** (verbatim from `UpdateModelAnimationVertexBuffers`). The mesh handed to
  the GPU is already in its final animated pose.
- Rendering then runs whatever shader is on `material.shader`. That shader sees
  `vertexPosition` / `vertexNormal` already posed. It needs *zero* bone
  knowledge — no `boneMatrices` uniform, no bone attributes. It's an ordinary
  static-mesh material shader as far as it's concerned.
- Therefore a custom shader that does Blinn-Phong/PBR lighting + normal mapping
  drops straight onto the CPU-skinned marine today. Assign it with
  `model.materials[i].shader = myLitShader` (in pyray:
  `model.materials[0].shader = my_shader`), wire the standard raylib locs
  (`SHADER_LOC_MATRIX_MVP`, `SHADER_LOC_MATRIX_MODEL`,
  `SHADER_LOC_MATRIX_NORMAL`, `SHADER_LOC_MAP_NORMAL`, light uniforms), done.

Confirmed correct for raylib. **GPU skinning is orthogonal**: it only moves the
per-vertex bone transform from CPU to GPU to lift the ~20-unit ceiling. It
changes *where* the vertex is posed, not *how* it is lit.

**Caveats to carry:**
- **Normals: fine.** raylib's CPU skinner re-transforms normals by the
  inverse-transpose of the bone matrix and re-uploads them
  (`animNormals`), so `vertexNormal` is correct in the posed frame.
- **Tangents: watch this.** raylib's CPU skinner does **not** re-skin tangents
  (there is no `animTangents`; only positions and normals are updated). The
  `TANGENT` attribute stays in bind pose. For tangent-space **normal mapping**
  on a heavily deformed limb this yields a slightly wrong tangent frame. Two
  clean mitigations: (a) reconstruct the TBN per-fragment from screen-space
  derivatives of position and UV (no vertex tangent needed), or (b) accept the
  minor error — for stylized marines at gameplay distance it's usually
  invisible. Under *GPU* skinning the same caveat applies unless the skinning
  `.vs` also skins the tangent, which the stock example does not.
- When you later flip to GPU skinning, the lit shader must move its skinning
  into the vertex stage (skin position+normal there, as in §1) — but the
  fragment/lighting half is unchanged. So the lit shader you write now is ~90%
  reusable; only the vertex preamble changes.

---

## 4. Recommendation for Breach

**Do the custom lit + normal-map shader now on top of CPU skinning. Defer GPU
skinning behind the existing `_draw_one` swap seam.** Rationale:

1. **The lit shader is unblocked and independent (§3).** It delivers the visual
   win (lighting, normal maps) immediately and carries no GPU-skinning risk. At
   "tens of units" the CPU-skinning cost is real but not yet the wall.

2. **GPU skinning is not a "write a shader" task with this wheel — it's a
   "rebuild the binding" task (§2).** The blocker is `SUPPORT_GPU_SKINNING=0` in
   the shipped static lib, not a missing Python function. Doing it *without*
   rebuilding means hand-reimplementing the bone-matrix loop and hand-managing
   VBO/VAO state to avoid a double transform — fragile, driver-sensitive, and
   exactly the plumbing the compile flag provides for free. That's a poor thing
   to entangle with a feel-facing shader change.

3. **When the unit ceiling actually bites, take the clean path:** build a custom
   raylib-python-cffi with `-DSUPPORT_GPU_SKINNING=ON` (also bump
   `MAX_BONE_NUM`/uniform limits as needed; the PR author notes 128 bones was
   safe, 256 exceeded his laptop's uniform space). Then the stock
   `UpdateModelAnimation` + `DrawModel` + a skinning-aware version of the lit
   vertex shader work with no Python VBO surgery, and CPU skinning
   auto-disables. Cost is a custom wheel build + CI packaging, not per-frame
   Python complexity.

4. **Keep the seam.** Because the fragment/lighting half of the shader is
   identical between CPU- and GPU-skinned paths, structure the lit shader so the
   only future change is the vertex preamble (raw vs. pre-posed vertices). The
   `_draw_one` swap seam is the right place to switch `update_model_animation`
   (CPU) for the GPU-attribute path later.

**Effort/risk read:** lit shader on CPU skinning = low risk, high immediate
value, feel-gated (HUMAN-TEST per project rules since it touches visuals). GPU
skinning now = medium-high effort (custom binding build OR fragile Python VBO
management), medium risk (double-transform, driver attribute-binding quirks,
uniform-count limits), and it only pays off past the ~20-unit ceiling — value
that isn't being realized yet. Sequence: **lit shader now, GPU skinning when
counts demand it, via a `SUPPORT_GPU_SKINNING` rebuild.**

### Uncertainties to verify before coding
- Exact struct layout in *this* wheel: is `boneMatrices`/`boneCount` on `Mesh`
  or `Model.skeleton`? Introspect via ffi in a REPL (the PR #4321 vs redesign
  split is real and version-dependent).
- Confirm the wheel truly built with `SUPPORT_GPU_SKINNING=0` by checking
  whether a rigged mesh's `boneIndices` VBO id in `mesh.vboId[6]` is 0/unset
  after `LoadModel` — cheapest empirical test of the whole §2 finding.
- Confirm `rl_set_uniform_matrices` and the `rl_*` vertex-buffer calls are
  exposed under those names in the pyray/raylib module (they mirror rlgl.h and
  should be, but verify).
