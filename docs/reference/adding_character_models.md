# Adding character models to Breach — a mini-course

*A practical guide for Erik (and anyone — hi Idir) extending Breach's library of
3D character models. Read this before pulling a model from Meshy, Mixamo, a
marketplace, or generating one with AI. Written 2026-07-21, at the close of the
first lit-marine arc; the concrete constraints below come from the pipeline we
actually built (`renderer/marine_shader.py`, `renderer/unit_model_renderer.py`).*

Breach is an ML/RL project wearing a game's clothes. Character models are
**render-only** — they never touch the simulation, determinism, or training.
But the *skeleton* they hang on is shared with the ML animation track, so the
one rule that governs everything here is:

> **A new model is a new SKIN on an existing SKELETON — not a new skeleton.**
> New skeleton only when the *body plan* is genuinely new (see §1).

---

## 0. The 30-second version (the checklist)

1. **Does it need a new skeleton?** Apply the topology test (§1). Almost always: no.
2. **Cheapest path — re-skin our own model.** If you just want a new *look* for an
   existing body plan, **texture our existing rigged model** (e.g. Meshy texturing
   mode) instead of importing a foreign mesh. This keeps our skeleton + all our
   animations and sidesteps retargeting completely (§3a). Prefer this.
3. **New mesh → retarget onto our skeleton.** If you must import a foreign mesh,
   it arrives with its own (or no) skeleton; you **re-rig it to our canonical
   skeleton** (§3b) so our clips and ML controllers drive it.
4. **Meet the technical constraints** (§4): glTF/GLB, low-poly, clean rig, UVs,
   sane scale/orientation, albedo sRGB / normal-map linear.
5. **Name the animation clips to match our `CLIP_MAP`** (§5).
6. **Record the source + license** in the asset folder's `LICENSE.txt` (§6) — and
   mind the ML-training clause.
7. **Verify** it loads, animates, and renders lit at real level size (§7).

---

## 1. First decision: skeleton or skin?

The expensive, rarely-taken branch is **adding a skeleton**. Take it only when the
creature's **movement topology** is new — not when the creature is merely new.

**The topology test — does it move on a fundamentally different limb layout?**
- Same layout → **same skeleton, new skin.** Marine and zombie share the humanoid
  skeleton; a hulking brute and a scrawny civilian share it too (differ by
  bone-scaled proportions). This is the common case.
- Different layout → **genuinely new skeleton.** Bipedal vs quadruped vs radial
  many-legged vs boneless-tentacled vs serpentine each need their own rig, because
  the IK/gait math and bone graph differ.

**Why we stay stingy with skeletons (it pays double):**
- Each skeleton needs its **own animation content** (you can't share a biped's
  walk with a spider), AND
- each skeleton is roughly **one ML motion-controller to train** (a humanoid
  policy doesn't transfer to a spider). So every skeleton is a content *and* a
  compute cost.

**Where variety comes from instead (cheap — be generous here):** skin/texture
swaps, bone-scale proportions, gait parameters, and pose libraries — the
per-unit *visual-profile* layer. Add these freely.

**Target skeleton set for Breach's bestiary (~4–6, not 1, not 20):** humanoid
(marines/zombies/civilians) · quadruped (dino/dog) · radial many-legged
(spider/woodlouse — ideally one parametric "N-legged" rig) · tentacled/soft
(octopus) · serpentine (snakes/worms). Do **not** cram a dissimilar body plan onto
an existing rig to save a slot — that's a false economy that fights the animation.
A single über-skeleton is a curiosity, not a goal.

**A new skeleton is a project, not a task** — it needs a rig, an animation set, a
gait/ML controller, and integration. Get explicit sign-off before starting one.

---

## 2. The two ways to get a "new model", ranked

### (a) BEST: re-skin our existing rigged model (no retargeting)
Take a model we already have rigged to our canonical skeleton and give it a new
**texture** (albedo + optional normal map). AI *mesh-texturing* tools (Meshy
texturing mode, etc.) paint new maps onto the model's existing UVs from a prompt.
You keep our skeleton, our animation clips, and our ML controller for free — the
new skin just drops into the pipeline. **This is the path for most new looks,
including variations and the "bloodied zombie" skin.**

### (b) When you need a genuinely new mesh: import + retarget
A foreign mesh (marketplace, or AI text-to-3D) comes with its own skeleton or
none. You must **re-attach it to our canonical skeleton** — see §3. More work;
only do it when re-skinning our model can't give you the shape you need.

---

## 3. Retargeting — the thing to get right (and the Meshy trap)

**The trap:** AI/marketplace models ship a *unique per-model skeleton*. We cannot
use that skeleton — our animations and ML controllers are bound to *ours*. So an
imported mesh must be bound to our canonical skeleton.

### 3a. Avoid it when you can
Re-skinning our own model (§2a) needs *zero* retargeting. Always check whether
that gets you there first.

### 3b. When you must retarget — the routes
- **Auto-rig a bare humanoid mesh to a standard skeleton.** Tools like **Mixamo**
  auto-rig any humanoid mesh from ~6 marker placements; **Blender Rigify** and
  similar do the same. Works well for standard bipeds.
- **Retarget an already-rigged model's motion onto our skeleton** via bone-name
  mapping (Blender retargeting add-ons, Rokoko, etc.).
- **Exotic morphologies (spider/octopus/worm) are much harder** — auto-riggers are
  humanoid-centric; expect manual weight-painting.

### 3c. Open question — how automatable is this? (TO INVESTIGATE)
For standard humanoids, auto-rig/retarget is fast and largely automated. For our
exotic skeletons it may be manual and laborious. **Before we lean on imported
meshes, run a real test:** take one Meshy/marketplace humanoid, retarget it onto
our humanoid skeleton, and time it / see if it can be scripted (Blender headless
+ an auto-rig step). Until that's answered, prefer §2a (re-skin our model).

---

## 4. Hard technical constraints (the pipeline requirements)

These come from what the renderer actually accepts today.

- **Format:** **glTF / GLB** (raylib loads it natively, incl. skeletal animation).
  FBX/other → convert to glTF. `.gltf`+`.bin`+textures, or a single `.glb`.
- **One clean skeleton per file, low bone count** (~30–60 for a humanoid). raylib's
  glTF *animation* importer has known bugs on exotic/multi-mesh/scale-animated rigs
  — **test every asset**; a clean single-skeleton humanoid is the safe case.
- **Low-poly mesh** (~1–5k tris). CPU-skinning ceiling is ~20 animated units until
  we add GPU skinning, so keep meshes light. Multi-mesh models are fine but the
  material shader must be set on **every sub-mesh** (our current marine has 2
  meshes / materials 1 & 2 — material 0 is dead).
- **UVs required** if you want textures/normal maps (the shader samples them).
- **Textures:** **albedo is sRGB** (a normal PNG); **normal maps are linear**
  (never sRGB-decoded) and mind the **Y-sign** (OpenGL vs DirectX — we expose
  `u_normal_y_sign` / the `H` key). Our shader takes albedo in the ALBEDO slot and
  a normal map in the **ROUGHNESS slot (sampler `texture3`)** — the light field
  occupies METALNESS+NORMAL. Drop a normal map over
  `assets/models/marine/marine_normal_PLACEHOLDER.png` and flip
  `MARINE_USE_NORMAL_DEFAULT` to enable it. **No texture is also fine** (flat
  material colour + our group tint).
- **Top-down readability:** we see mostly the *top* surface. Bake **ambient
  occlusion into the albedo** and keep a readable **silhouette** — these matter
  more than fine detail. A normal map helps catch the scene's side-lights.
- **Scale & orientation:** glTF **Y-up**; a consistent forward axis. We normalize
  on-screen size via `_SCALE_TILES_TALL` and facing via `_YAW_OFFSET_DEG`
  (calibrated), but a sanely-oriented model makes that trivial.
- **Render-only, always:** no gameplay/sim data on the mesh. Model + animation
  state lives in the renderer (`UnitModelRenderer`), keyed by `unit.id`, never on
  `Unit` — so it stays out of the synced sim/digest and is skipped in headless
  training.

---

## 5. Animation constraints

- Clips must be authored/retargeted **for our skeleton** and **named to match our
  `CLIP_MAP`** (`idle`/`walk`/`fire`/`dead` today; `limp`/`wounded`/… later). A
  missing name falls back to idle.
- A **shared skeleton means one animation library serves every skin on it** — like
  Quaternius' Universal Animation Library does for our humanoids. This is the big
  payoff of skeleton parsimony: source/retarget the clip set once, reuse across all
  humanoid skins.
- **For the ML track:** the skeleton's bone structure defines the controller's
  action space, so keep it consistent — one trained controller then drives every
  skin on that skeleton.

---

## 6. Licensing & provenance (record it every time)

Put a `LICENSE.txt` in the asset's folder naming the **source + license**. Known
cases:
- **CC0** (Quaternius, Kenney) — frictionless: commercial OK, no attribution,
  ML-training OK.
- **Mixamo** — royalty-free for games, **but forbids using its assets to train ML
  models.** Fine for our render-only marines — but keep such assets strictly
  render-side; never feed rendered frames of them into model training.
- **AI-generated** (Meshy etc.) — check the tool's **commercial-use + ownership**
  terms per plan; they vary.
- **Sketchfab CC-BY** — needs attribution.

**The ML-training clause is a real constraint given our identity:** if an asset
forbids ML-training use, it must never enter a training-data path. Render-only
keeps us safe; just don't blur that line.

---

## 7. Verify before you commit it

Mirror the harness pattern we used for the marines:
1. **Load-verify:** confirm the model loads, reports its bones + clips, and is
   rigged (`scratchpad/introspect_model.py` is a template).
2. **Render-verify at REAL level dimensions** (not a tiny synthetic scene — that
   hid two bugs this arc): render it in the actual world at real size, lit, and eyeball
   it beside the ship (`scratchpad/verify_p1_real.py` is a template).
3. **Regression:** the rest of the game is unchanged.
Then add the asset + its `LICENSE.txt` with explicit `git add` paths (never
`git add -A`).

---

## 8. Red-flags / gotchas checklist (skim before importing)

- [ ] Model ships a **unique skeleton** → must retarget to ours (§3), or re-skin
      our model instead (§2a).
- [ ] **raylib glTF import** quirks on non-trivial rigs → test the actual file.
- [ ] **Multi-mesh** model → set the shader on *all* sub-mesh materials.
- [ ] **Normal-map Y-sign** and **sRGB/linear** (albedo sRGB, normal linear).
- [ ] **Non-humanoid auto-rigging is hard** — budget manual work.
- [ ] **Bone count / tri count** sane (CPU-skinning ceiling ~20 units for now).
- [ ] **Licence** recorded; **ML-training clause** respected (render-only).
- [ ] **A new skeleton?** — stop, get sign-off; it's a project, not a task (§1).

---

*Related: `docs/procedural_animation_brainstorm.md` (the skeleton/gait design),
`docs/marine_shader_foundation_design_2026-07-20.md` (the lit-shader pipeline these
constraints come from), `docs/research/ml_animation_litsearch_2026-07-20.md` (the ML
animation track this all feeds). Backlog: `docs/TODO.md` "Animation / character-render
track".*
