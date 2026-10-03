# Agent silhouette study — 2026-10-03

Rough silhouette and major forms for Erik's visual review, not a finished or runtime-ready character. Authored editable mesh surfaces from the supplied concept; separate armour, gloves, boots, helmet, duffel and weapon. No downloaded model, generated image, external service, new skeleton or runtime code.

Reference: `../sleek-agent-with-bag.png`, supplied and approved by Erik. SHA256: `65507599E4730472822BFE5ED607D350D9C9E0834115CF713279570A8DD4395B`. Reference authorship/licensing was not supplied; this note does not assign a third-party licence to it. Original concept remains unaltered.

Canonical compatibility target: existing Quaternius Universal Animation Library, `assets/models/marine/AnimationLibrary_Godot_Standard.gltf`, with repository CC0 provenance in that directory. Imported in a disposable Blender session, not edited or copied into the new character. `source/canonical_rig_inspection.json` records its 53 bone names, parents, local rest matrices, heads/tails and armature world matrix. The inspected mannequin has 8,547 vertices and 13,743 imported triangle polygons; its T-pose width is not body width. The 42-vertex `Icosphere` is an importer auxiliary object excluded from the assessment preview. `previews/canonical_base_assessment.png` shows the actual rest mesh.

The reusable anatomy is broadly humanoid, but the broad segmented chest and muscular upper arms do not match the approved fitted slim female silhouette. Texture-only reskinning cannot change those forms or add the sealed helmet. New ring surfaces and tailored faceted plates were authored instead. Later binding must preserve the existing hierarchy and rest transforms; names alone prove nothing.

`source/agent_silhouette.blend` is editable source. `scripts/build_agent.py` reproduces the scene and six PNGs, with named adjustable materials (off-white, charcoal, rubber/webbing, visor, teal and hardware), bevel modifiers, independent equipment collections, and removable review studio. All shaders use ordinary surface shading without emission. The suit/limbs use editable ring strips. Surfaces overlap at joints: this is a review construction, not final welded deformation topology. Helmet visor, respirator, sealed collar, shoulder/chest/forearm/lower-leg armour and shaped boot surfaces establish major forms. Unseen back armour and calf guards are provisional design decisions. Fingers, bag and original compact bullpup housing remain placeholders.

`front`, `side`, `back`, `three_quarter`, `top_down_vertical` and separately labelled `top_down_tilted_readability` live in `previews/`. The vertical view looks exactly down Blender -Z with zero camera tilt and orthographic projection, corresponding to `renderer/lit3d.py`'s vertical runtime camera after Z-up to Y-up export. These are neutral Blender studio previews, not Breach shader screenshots. Colours and roughness require later actual-lighting review.

Reproduce from this worktree root using PowerShell:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python art/characters/agent/scripts/inspect_canonical.py
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python art/characters/agent/scripts/build_agent.py
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background art/characters/agent/source/agent_silhouette.blend --python art/characters/agent/scripts/verify_saved.py
```

Blender 4.5.10 LTS bundled Python was used. A sandbox warning says Blender cannot update its user extension cache; successful saved files and renders do not depend on that cache. Mesh counts live in `source/mesh_statistics.json`, including base and evaluated triangles; no final export ceiling is imposed. No texture UVs, skin weights, rig, animation or runtime export is claimed. No game tests are needed for offline-only asset work. Actual glTF loader, hierarchy/rest/weights and shader verification await the later rigged asset. This branch is based on older `main`; before future integration, recheck the current runtime branch's pipeline rather than treating July notes as permanent constraints.

Next: Erik reviews proportions and major armour silhouette. Revise those forms first; then agree deformation topology and binding to canonical rig. Animation production remains deferred. HUMAN-TEST gate: do not merge before visual review.
