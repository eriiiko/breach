# Lighting a 3D character + normal mapping in raylib, combined with skinning — one shader

Research digest for Breach. Date: 2026-07-20. Focus: the **fragment lighting +
normal-map technique** and how it must coexist with GPU skinning in a single
shader, plus what a true 90° top-down view demands. A sibling digest covers the
skinning mechanics; this one owns the lighting/normal-map side.

All source references are raylib `master` (raylib 5.5-dev era). Line numbers cited
below are from the state of the repo on 2026-07-20 and may drift.

---

## 0. TL;DR recipe (the minimal correct path)

1. Mesh must carry **tangents** (glTF `TANGENT` attribute, or `GenMeshTangents()`).
   raylib auto-binds `in vec4 vertexTangent` to attribute location 4 for any custom
   shader — you get tangents for free if the mesh has them.
2. Bind the normal map to `model.materials[i].maps[MATERIAL_MAP_NORMAL].texture`;
   sample it in the fragment shader as `sampler2D normalMap` (or whatever name you
   wire to `SHADER_LOC_MAP_NORMAL`).
3. In the **vertex shader**: skin position **and** normal **and** tangent by the
   bone matrices; build a world-space TBN; pass world-space position + TBN + UV out.
4. In the **fragment shader**: sample normal map, `N = normalize(sampledNormal*TBN)`
   (tangent→world), then loop up to 4 lights doing `max(0,N·L)` diffuse + Blinn-Phong
   spec + ambient. Gamma-correct at the end.
5. The two classic failures are: (a) **not skinning the tangent/normal** (lighting
   swims on animated limbs), and (b) **normal-map Y-sign / handedness** (bumps light
   from the wrong side). Breach already exposes `u_normal_y_sign` for exactly (b).

---

## 1. raylib's canonical lighting shaders

### 1a. `rlights.h` — the up-to-4-light Blinn-Phong helper

`examples/models/rlights.h` and `examples/shaders/rlights.h` are the single-header
convenience for the built-in `lighting.vs`/`lighting.fs` pair. It is **C-side glue**,
not a shader — it just locates and pushes uniforms.

```c
#define MAX_LIGHTS  4          // hard cap the fragment shader loops over

typedef enum { LIGHT_DIRECTIONAL = 0, LIGHT_POINT = 1 } LightType;

typedef struct {
    int type;                  // directional | point
    bool enabled;
    Vector3 position;
    Vector3 target;            // for point lights, target is ignored
    Color color;
    float attenuation;         // present in struct; not used by stock fragment math
    // cached shader-uniform locations:
    int enabledLoc, typeLoc, positionLoc, targetLoc, colorLoc;
} Light;

Light CreateLight(int type, Vector3 pos, Vector3 target, Color color, Shader shader);
void  UpdateLightValues(Shader shader, Light light);
```

- `CreateLight()` calls `GetShaderLocation()` for each field using the indexed-array
  naming convention `lights[i].enabled`, `lights[i].type`, `lights[i].position`,
  `lights[i].target`, `lights[i].color`. A file-scope counter (`lightsCount`) gives
  each light its `i`. Then it calls `UpdateLightValues()` once.
- `UpdateLightValues()` pushes every field with `SetShaderValue(...)`, converting
  `Color` (0–255 bytes) to normalized `vec4` floats and splitting `Vector3` into a
  `float[3]`.
- **Per-frame:** you set `lights[i].position`/`color`/`enabled` on the CPU `Light`
  struct then call `UpdateLightValues(shader, light)` again. Nothing is automatic.
- Two extra uniforms live on the *shader*, not the light: `ambient` (`vec4`) and
  `viewPos` (`vec3`, the camera position, needed for the specular view vector). You
  set these yourself each frame — `viewPos` whenever the camera moves; `ambient`
  once.

`Bigfoot71/rlights` is a more capable community fork (spotlights, attenuation,
shadow support) if the stock 4-light cap becomes limiting — worth knowing it exists,
but the stock header is the reference.

### 1b. The stock fragment math — `lighting.fs`

This is the exact math Breach should port for a Blinn-Phong marine. Structure:

```glsl
#define MAX_LIGHTS 4
#define LIGHT_DIRECTIONAL 0
#define LIGHT_POINT 1
struct Light { int enabled; int type; vec3 position; vec3 target; vec4 color; };
uniform Light lights[MAX_LIGHTS];
uniform vec4  ambient;
uniform vec3  viewPos;

vec3 normal = normalize(fragNormal);
vec3 viewD  = normalize(viewPos - fragPosition);
vec3 lightDot = vec3(0.0), specular = vec3(0.0);

for (int i = 0; i < MAX_LIGHTS; i++) {
    if (lights[i].enabled == 1) {
        vec3 light;
        if (lights[i].type == LIGHT_DIRECTIONAL)
            light = -normalize(lights[i].target - lights[i].position);
        if (lights[i].type == LIGHT_POINT)
            light =  normalize(lights[i].position - fragPosition);

        float NdotL = max(dot(normal, light), 0.0);      // Lambert diffuse
        lightDot += lights[i].color.rgb * NdotL;

        float specCo = 0.0;
        if (NdotL > 0.0)
            specCo = pow(max(0.0, dot(viewD, reflect(-light, normal))), 16.0); // 16 = shininess
        specular += specCo;
    }
}
finalColor  = texelColor * ((tint + vec4(specular,1.0)) * vec4(lightDot,1.0));
finalColor += texelColor * (ambient/10.0) * tint;
finalColor  = pow(finalColor, vec4(1.0/2.2));            // gamma correction
```

Notes / gotchas worth carrying:

- The stock spec term is **Phong** (`reflect(-light,normal)` dotted with view), not
  true Blinn-Phong (half-vector). `pow(...,16.0)` hard-codes shininess. For a marine
  you'll likely want a half-vector form `H=normalize(light+viewD); pow(max(0,dot(N,H)),shin)`
  and a shininess uniform.
- Ambient is added as `ambient/10.0` — a magic scale. Diffuse and ambient are both
  multiplied by `texelColor`, so the albedo controls both. Fine for a start.
- The final `pow(color, 1/2.2)` assumes the shader worked in linear space and the
  framebuffer is not already sRGB. If your textures are sRGB PNGs you technically
  should linearize on sample too; the stock example does not, which is a mild
  correctness sin (see §5 gamma).
- The vertex side (`lighting.vs`) does `fragPosition = matModel*vertexPosition`
  (world space) and `fragNormal = matNormal*vertexNormal`. **Key fact:** raylib sets
  `matNormal = transpose(inverse(matModel))` in `DrawMesh` (rmodels.c ~L1538), i.e.
  the **world-space** normal matrix — despite a stale comment in rlgl.h claiming it is
  `modelView`. So lighting here is done in **world space**, and `viewPos`/light
  positions must also be world-space. Good; consistent.

### 1c. The PBR example — `shaders_basic_pbr` (`pbr.vs`/`pbr.fs`)

`examples/shaders/shaders_basic_pbr.c` demonstrates a full metallic-roughness PBR
path. Same 4-light struct but with an added `float intensity` per light and a
`numOfLights` count uniform (loops `for i < numOfLights`). It samples:
`albedoMap`, `mraMap` (metalness/roughness/AO packed in R/G/B), `normalMap`,
`emissiveMap`, plus scalar fallbacks (`metallicValue`, `roughnessValue`, `aoValue`)
and `useTexNormal`/`useTexMRA`/... toggles. Core BRDF is Cook-Torrance:
`SchlickFresnel`, `GgxDistribution` (GGX/Trowbridge-Reitz NDF), `GeomSmith`
(Smith geometry). This is heavier than a top-down marine needs — Blinn-Phong from
§1b is the right altitude for Breach — but the PBR **normal-map handling in `pbr.vs`
is the cleanest reference for the TBN construction** (see §2), so borrow that even if
you skip the BRDF.

---

## 2. Normal mapping on a character

### 2a. Tangents are mandatory

A normal map stores per-texel normals in **tangent space** (the surface's local
frame, where +Z is "out of the surface"). To use them you must transform them into
the space you light in (world space, for raylib). That transform needs a **TBN**
basis per fragment: Tangent, Bitangent(=binormal), Normal.

- **Where tangents come from:** glTF meshes usually ship a `TANGENT` attribute; raylib
  loads it into `mesh.tangents` (XYZW, 4 floats/vertex). If missing, call
  `GenMeshTangents(&mesh)` (raylib.h L1629) to compute them from positions + UVs.
  The `.w` component encodes **handedness** (±1) — critical, see §5.
- **Binding:** raylib's `rlLoadVertexArray` auto-binds `vertexTangent` to attribute
  **location 4** (`RL_DEFAULT_SHADER_ATTRIB_LOCATION_TANGENT`, rlgl.h L343/4355), so
  any custom shader that declares `in vec4 vertexTangent;` receives it with no manual
  `glBindAttribLocation`. Same automatic treatment as position/normal/uv.
- **Texture slot:** put the normal texture in
  `material.maps[MATERIAL_MAP_NORMAL].texture` (raylib.h L773). Wire the sampler
  uniform to `SHADER_LOC_MAP_NORMAL` (L808) so raylib binds it to the right texture
  unit at draw time. (`MATERIAL_MAP_DIFFUSE`/`_ALBEDO` → `SHADER_LOC_MAP_ALBEDO`,
  bound to `texture0`, is the albedo equivalent.)

### 2b. Building and using the TBN (from `pbr.vs` / `normalmap.vs`)

Vertex shader — build world-space TBN and pass it out:

```glsl
in vec3 vertexNormal;
in vec4 vertexTangent;                       // .xyz tangent, .w handedness
uniform mat4 matModel;
out mat3 TBN;

vec3 vertexBinormal = cross(vertexNormal, vertexTangent.xyz) * vertexTangent.w;
mat3 normalMatrix   = transpose(inverse(mat3(matModel)));

vec3 N = normalize(normalMatrix * vertexNormal);
vec3 T = normalize(normalMatrix * vertexTangent.xyz);
T = normalize(T - dot(T, N)*N);              // Gram-Schmidt re-orthogonalize
vec3 B = cross(N, T);                        // rebuild B orthonormal (drops handedness!)
TBN = transpose(mat3(T, B, N));              // note: transpose → maps tangent→world via N*TBN
```

Fragment shader — sample and transform:

```glsl
uniform sampler2D normalMap;
in mat3 TBN;

vec3 n = texture(normalMap, fragTexCoord).rgb;
n = normalize(n * 2.0 - 1.0);                // unpack 0..1 → -1..1
n = normalize(n * TBN);                      // tangent-space → world-space
// ... use n as the shading normal in the light loop
```

Two things to flag about the **stock** raylib TBN:

1. It builds `TBN = transpose(mat3(T,B,N))` and then does `n * TBN` (vector-times-
   matrix). This is the transpose of the more common `TBN * n` convention. It is
   internally consistent — just don't mix conventions when porting Breach's 2D
   normal-map math (which uses its own `u_normal_y_sign` flip).
2. `B = cross(N, T)` **discards** `vertexTangent.w` handedness after using it once to
   compute `vertexBinormal` (which is then overwritten). On meshes with mirrored UVs
   this is a bug source; the correct robust form keeps `B = cross(N,T) * vertexTangent.w`.
   Watch this if the marine's UVs are mirrored (common on symmetric humanoids!).

---

## 3. The skinning ⊗ normal-map interaction (the crux)

**GPU skinning transforms geometry in the vertex shader by a weighted sum of bone
matrices. If you skin only the position, the normal and tangent stay in bind pose,
so lighting on a bent limb is wrong** — the surface moves but its lighting frame does
not. You must skin the **normal and the tangent by the same bone matrices** as the
position.

### 3a. What the stock skinning shader does — and doesn't

`examples/models/resources/shaders/glsl330/skinning.vs` skins position and normal:

```glsl
#define MAX_BONE_NUM 128
in vec4 vertexBoneIndices;
in vec4 vertexBoneWeights;
uniform mat4 boneMatrices[MAX_BONE_NUM];    // uniform name "boneMatrices"

// weighted position:
vec4 skinnedPosition = w.x*(B[i0]*vec4(pos,1)) + w.y*(B[i1]*vec4(pos,1)) + ...;
// weighted normal (w=0 so translation is ignored):
vec4 skinnedNormal   = w.x*(B[i0]*vec4(nrm,0)) + w.y*(B[i1]*vec4(nrm,0)) + ...;
fragNormal = normalize(vec3(matNormal * skinnedNormal));
gl_Position = mvp * skinnedPosition;
```

**Its stock `skinning.fs` does NO lighting** — it just samples `texture0*colDiffuse*fragColor`.
And it **does not skin a tangent** — there is no `vertexTangent` input. So the stock
pair is *not* usable as-is for a normal-mapped, lit marine. You must extend it (§3c).

Subtlety: skinning the normal with `B*vec4(n,0)` uses the bone matrix directly, not
its inverse-transpose. That is only exactly correct for rigid (rotation+translation,
no non-uniform scale) bones — which is the normal case for character rigs, so it is
fine. If any bone has non-uniform scale you would need the inverse-transpose per bone.

### 3b. How the bone matrices actually reach the shader

You do **not** upload `boneMatrices` yourself. `DrawMesh` (rmodels.c ~L3954) pushes
them automatically:

```c
if (mat.shader.locs[SHADER_LOC_MATRIX_BONETRANSFORMS] != -1 && model.boneMatrices)
    rlSetUniformMatrices(mat.shader.locs[SHADER_LOC_MATRIX_BONETRANSFORMS],
                         model.boneMatrices, model.skeleton.boneCount);
```

`SHADER_LOC_MATRIX_BONETRANSFORMS` (raylib.h L819) maps to the uniform named
`boneMatrices` (rlgl.h L1018). `vertexBoneIndices`/`vertexBoneWeights` auto-bind to
attribute locations 7/8. The per-frame flow is:

```c
model.materials[i].shader = skinningShader;              // assign once
...
UpdateModelAnimationBones(model, anim, frame);           // fills model.boneMatrices (GPU path)
DrawModel(model, pos, scale, WHITE);                     // uploads boneMatrices, draws
```

`UpdateModelAnimationBones()` (or `UpdateModelAnimation()` when built with
`SUPPORT_GPU_SKINNING`) computes `boneMatrices[b] = invBindPose * currentPose`
(rmodels.c L2351/2437). **GPU skinning requires raylib built with `SUPPORT_GPU_SKINNING`**;
otherwise raylib silently falls back to CPU skinning.

### 3c. CPU skinning: what raylib re-transforms (and what it forgets)

If you use CPU skinning (`UpdateModelAnimation` without the GPU flag), raylib rewrites
`mesh.animVertices` and `mesh.animNormals` on the CPU each frame (rmodels.c L2484-2540):

- Positions: weighted `Vector3Transform(v, boneMatrix)`.
- Normals: weighted `Vector3Transform(n, transpose(invert(boneMatrix)))` — correct
  inverse-transpose. **Then it re-uploads only the position and normal VBOs**
  (L2539-2540).
- **Tangents are NOT touched.** There is no `animTangents` array anywhere in rmodels.c.

**Therefore CPU skinning breaks normal mapping on animated limbs** just as surely as
un-skinned GPU normals would — the tangent frame stays in bind pose. If Breach wants
normal-mapped *and* animated marines, the clean answer is **GPU skinning with a custom
shader that skins position + normal + tangent together** (§3d). CPU skinning is only
"safe" for normal mapping if the mesh is static, or if you accept subtly wrong bump
lighting on moving parts.

### 3d. The combined shader (skin + TBN), sketch

Vertex — skin P, N, T by the bones, then build the TBN from the *skinned* N/T:

```glsl
#version 330
#define MAX_BONE_NUM 128
in vec3 vertexPosition; in vec2 vertexTexCoord; in vec3 vertexNormal;
in vec4 vertexTangent;  in vec4 vertexBoneIndices; in vec4 vertexBoneWeights;
uniform mat4 mvp, matModel, matNormal;
uniform mat4 boneMatrices[MAX_BONE_NUM];
out vec3 fragPosition; out vec2 fragTexCoord; out mat3 TBN;

void main() {
    ivec4 bi = ivec4(vertexBoneIndices);
    vec4 w   = vertexBoneWeights;
    mat4 skin = w.x*boneMatrices[bi.x] + w.y*boneMatrices[bi.y]
              + w.z*boneMatrices[bi.z] + w.w*boneMatrices[bi.w];   // blended bone matrix

    vec4 sp = skin * vec4(vertexPosition, 1.0);
    vec3 sn = mat3(skin) * vertexNormal;          // rigid bones: mat3 ok
    vec3 st = mat3(skin) * vertexTangent.xyz;

    mat3 nm = transpose(inverse(mat3(matModel)));
    vec3 N = normalize(nm * sn);
    vec3 T = normalize(nm * st);
    T = normalize(T - dot(T,N)*N);
    vec3 B = cross(N, T) * vertexTangent.w;       // keep handedness!
    TBN = mat3(T, B, N);                           // world-from-tangent

    fragPosition = vec3(matModel * sp);
    fragTexCoord = vertexTexCoord;
    gl_Position  = mvp * sp;
}
```
(Blending the *matrix* once, then applying it to P/N/T, is equivalent to blending the
three transformed vectors and a little cheaper — either is fine.)

Fragment — sample normal map, transform, run the §1b light loop:

```glsl
vec3 n = texture(normalMap, fragTexCoord).rgb;
n = n * 2.0 - 1.0;
n.y *= u_normal_y_sign;                 // Breach's convention (see §5)
vec3 N = normalize(TBN * n);            // tangent → world (matches mat3(T,B,N) above)
// ... loop lights exactly as §1b, using N and normalize(viewPos - fragPosition)
```

### 3e. Material/shader binding checklist (C side)

- `model.materials[i].shader = marineShader;` for each material that uses skinning.
- `marineShader.locs[SHADER_LOC_MAP_NORMAL] = GetShaderLocation(s, "normalMap");`
  and set `material.maps[MATERIAL_MAP_NORMAL].texture`.
- `SHADER_LOC_MATRIX_MODEL` (`matModel`), `_MATRIX_NORMAL` (`matNormal`),
  `_MATRIX_MVP` (`mvp`) auto-locate under those standard names — raylib fills them
  during `LoadShader`. `SHADER_LOC_MATRIX_BONETRANSFORMS` likewise (`boneMatrices`).
- Set `viewPos`, `ambient`, and each `lights[i].*` per §1.

---

## 4. Top-down 90° specifics (Breach's actual camera)

Breach looks **straight down** with an orthographic camera. This is the hardest
lighting case for reading 3D form, and it changes the priorities:

- **From directly overhead you see mostly TOP surfaces** (shoulders, helmet, backpack).
  A single **overhead** light (light directly above, `L ≈ N` for those top faces)
  makes `N·L ≈ 1` everywhere on top → **flat, no shape**. Overhead-only lighting is
  the worst choice for a top-down character.
- **Angle the key light.** Put the main directional light **off-vertical** (e.g. 30–60°
  from straight down, biased toward one side/"north-west"). Now `N·L` varies across the
  curved top surfaces and the character gets a light side and a shaded side — the single
  biggest readability win. This is a scene-authoring decision, not a shader one, but the
  shader must support directional lights (it does, §1b).
- **Fill + rim.** Add a dim **fill** light from the opposite side so shadowed faces
  don't go black, and consider a **rim/back** light to pop the silhouette against the
  floor. On a ~2–3 tile sprite the rim is what separates "a lit model" from "a token".
- **Normal maps do the heavy lifting here.** Because the mesh's own top surfaces are
  nearly flat to the camera, the *large-scale* `N·L` variation is small. The **normal
  map** re-introduces surface detail (armor plates, straps, muscle) whose micro-normals
  catch the angled key and the scene's side-lights — this is exactly why Breach wants
  normal-mapped marines rather than flat-lit ones. The angled key + normal map combo is
  the mechanism that makes a top-down character read as 3D.
- **Bake AO into the albedo/texture.** Ambient occlusion (crevices, under the arms,
  helmet rim) painted or baked into the albedo gives free depth cues that survive even
  under flat ambient, independent of light direction. Cheap and very effective top-down.
- **Match the scene's lights.** Breach's world already has per-tile `light_map` +
  side-light normal mapping on the 2D ship. Feed the *same* light positions/colors into
  the marine's `lights[]` so the marine is lit by the scene, not a separate rig — that
  is what makes it "part of the scene." A marine standing next to a light source should
  brighten on that side.
- **Scale sanity:** ~2k tris / ~53 bones over a 2–3 tile footprint means the marine is
  small on screen. Don't over-invest in high-frequency spec; a soft Blinn-Phong with
  modest shininess (16–32) plus a good normal map and one angled key reads best. Sharp
  speculars just sparkle-alias at that size.

---

## 5. How people get it wrong — and the minimal correct recipe

Common failure modes (each maps to a concrete check):

1. **Un-skinned normals/tangents.** Position skinned but N/T left in bind pose →
   lighting "swims" or stays static while limbs move. *Fix:* skin N and T with the
   same bone blend (§3d). Remember **CPU skinning silently does NOT skin tangents**
   (§3c) — GPU skinning is the clean route for normal-mapped animation.
2. **Wrong tangent handedness / dropped `.w`.** Stock raylib TBN rebuilds
   `B = cross(N,T)` and drops `vertexTangent.w`; on mirrored UVs half the model lights
   inverted. *Fix:* `B = cross(N,T) * vertexTangent.w`.
3. **Normal-map Y (green channel) sign — OpenGL vs DirectX convention.** Tools export
   normal maps with +Y-up (OpenGL) or +Y-down (DirectX). Wrong sign → bumps light from
   below instead of above; surfaces look inverted/embossed-wrong. *Fix:* flip `n.y`.
   **Breach already exposes `u_normal_y_sign`** for its 2D path — reuse the exact same
   uniform and convention on the marine so both agree.
4. **Gamma / sRGB.** Albedo/normal PNGs are usually sRGB-encoded. Doing math on sRGB
   values then `pow(1/2.2)` at the end double-darkens; and **normal maps must be treated
   as LINEAR data — never sRGB-decode a normal map** (its RGB are directions, not
   color). *Fix:* linearize *albedo* on sample (or use an sRGB texture format), keep
   normal/MRA maps linear, light in linear space, gamma-encode once at output. The stock
   `lighting.fs` skips input linearization — acceptable for a stylized look, but know it
   is a shortcut.
5. **Space mismatch.** Lighting in world space (raylib's `matNormal` is world-space,
   §1b) but feeding view-space light positions, or vice versa → lights appear to move
   with the camera. *Fix:* keep everything world-space; `viewPos` = camera world
   position; light positions = world.
6. **Forgetting `SUPPORT_GPU_SKINNING`.** Build raylib without the flag and your GPU
   skinning shader gets no bone deformation (raylib runs CPU skinning into
   `animVertices` instead and your `boneMatrices` uniform is never what you think).
   *Fix:* compile raylib with `SUPPORT_GPU_SKINNING` when going the GPU route.

**Minimal correct recipe:** glTF with tangents (or `GenMeshTangents`) → GPU skinning
raylib build → one custom shader that (vs) skins P/N/T by `boneMatrices` and emits
world-space position + TBN, (fs) samples `normalMap`, unpacks + Y-sign-flips + `TBN*n`
to world, then loops ≤4 lights with `max(0,N·L)` diffuse + half-vector spec + ambient,
gamma-encode once. Drive `lights[]`/`viewPos` from the same scene lights as the 2D
world, and place an **angled** key (not overhead) plus fill/rim for top-down readability.

---

## Sources

- raylib `lighting.vs`/`lighting.fs` + `rlights.h` (Blinn-Phong 4-light):
  https://github.com/raysan5/raylib/blob/master/examples/models/rlights.h ,
  https://github.com/raysan5/raylib/blob/master/examples/shaders/resources/shaders/glsl330/lighting.fs
- raylib basic PBR example (`pbr.vs`/`pbr.fs`, MATERIAL_MAP_NORMAL, MRA, GGX):
  https://github.com/raysan5/raylib/blob/master/examples/shaders/shaders_basic_pbr.c
- raylib normal-map example (`normalmap.vs`/`normalmap.fs`, TBN, useNormalMap):
  https://github.com/raysan5/raylib/blob/master/examples/shaders/resources/shaders/glsl330/normalmap.fs
- raylib GPU skinning example + `skinning.vs` (skins normal, no tangent, no lighting):
  https://github.com/raysan5/raylib/blob/master/examples/models/models_animation_gpu_skinning.c ,
  https://github.com/raysan5/raylib/blob/master/examples/models/resources/shaders/glsl330/skinning.vs
- raylib GPU skinning PR (Daniel Holden / orangeduck), boneMatrices uniform design:
  https://github.com/raysan5/raylib/pull/4321
- raylib source facts: `matNormal = transpose(inverse(matModel))`, CPU skinning
  transforms normals but not tangents, `boneMatrices` auto-upload —
  `src/rmodels.c` (DrawMesh ~L1538/L3954, CPU skin L2484-2540), `src/rlgl.h`
  (default attrib/uniform names L1006-1018), `src/raylib.h` (SHADER_LOC_* enums L773-819).
- Normal mapping / TBN background:
  http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/ ,
  https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
- `Bigfoot71/rlights` (extended lighting fork): https://github.com/Bigfoot71/rlights
