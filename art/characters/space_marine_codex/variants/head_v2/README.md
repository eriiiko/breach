# Space Marine (Codex): head v2

Completed 2026-10-05. This is an additional full-character source with a helmet-only edit. **The approved original is preserved**, available as a byte-identical [approved-v1 snapshot](../../versions/approved_v1/space_marine_codex_approved_v1.blend) and in its unchanged [original seven-view gallery](../../previews/index.html). Erik approved the original; the parent reviewer has inspected and accepted all five final variant views for delivery as an alternate. Choosing the variant and integrating either version remain separate decisions.

Open the [comparison gallery](previews/index.html), [comparison sheet](previews/comparison_heads.png), or [head-v2 Blender source](source/space_marine_codex_head_v2.blend). The gallery includes the supplied reference, matched original/new front, side, back and three-quarter close-ups, plus the new full-figure hero. All nine final model images are actual Cycles renders with unchanged studio lighting. Close-ups are 1600 x 1600; the full figure is 1600 x 2000, all at 144 samples. First-review images remain separately under `previews/review_1/`.

The compact shell has separate plates over a recessed substrate, longitudinal crown joints, a more projecting side-wrapped visor, a flatter lower optical edge with clipped rounded corners, a broader ivory chin/cheek frame and curved temple carriers. The original collar, body, equipment and pose are retained. One focused correction followed the first review; there was no further aesthetic pass.

The final shell is still a cleaner, simpler interpretation than the concept's panel construction. The approved body's stylized diagonal cloth creases are unchanged. This alternate does not supersede the approved source or claim exact concept reproduction.

Measured preservation and source validation:

| Check | Result |
|---|---:|
| Evaluated triangles, complete variant | **1,262,274** |
| New helmet/visor evaluated triangles | 137,192 |
| Unchanged non-head evaluated triangles | 1,125,082 |
| Evaluated vertices, complete variant | 634,849 |
| Non-head object fingerprints unchanged | 415 |
| Non-head material fingerprints unchanged | 12 |
| Frozen original files unchanged | 18, plus approved snapshot |
| Gallery links | 28 checked, none missing |

Only excessive sampling on the newly authored smooth head surfaces was reduced to reach this count; the body was not rebuilt or decimated. The source was reopened alongside the frozen original and independently compared: evaluated geometry, world transforms, material assignments and shader controls match outside the head. Studio lighting and color management also match. The render manifest hashes match the delivered sources. See [source validation](source/validation.json), [preservation fingerprints](source/preservation_fingerprints.json), [original hashes/provenance](../../versions/approved_v1/manifest.json) and [package validation](previews/package_validation.json).

The approved source/snapshot SHA-256 remains `75b74ff12e5d8073ca8e10253c57d21b8a5de31b28390773244f64db3c3140e9`. The new source SHA-256 is `28cd621070de9b18a700b43a277abd9e7a4fdf84becb23b469dbf539dd94cca0`. The reference is packed; no external image, missing resource, armature, live cloth modifier or external cache is required. The 234 triangles below 1e-12 square metres are unchanged from the original source. Layered components intentionally overlap; this is a high-detail editable source, not a watertight or deformation-ready game mesh.

The visor remains independently selectable as `VISOR GLASS / convex optical shield`, in `03 / Visor glass selection`, with the unique material `VISOR GLASS / olive optical coating`. Object/Material Index **1** identify only the glass. Its direct Principled settings are Roughness **0.14**, Coat Weight **0.52**, Coat Roughness **0.065**; there is no gloss override. The gasket and retaining frame use separate geometry/materials. This retains the documented input convention for the later charkit gloss workflow; no bake or engine export was performed.

Render the existing saved original and variant without rebuilding or saving either:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/variants/head_v2/scripts/head_variant.py' -- --render both
```

Rebuild only this variant from the frozen snapshot, render and reopen-audit it:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/variants/head_v2/scripts/head_variant.py' -- --build --render both --audit
```

`--render head_v2` renders only the saved variant, including hand edits. `--audit` reopens and validates it. `--draft` writes 1100-pixel comparisons to `previews/review_1/`; it would overwrite those initial-review images, so omit it when preserving the review history. Rebuilding overwrites only the variant source, never the approved original.

Recompose the gallery and labelled reference crops from existing images:

```powershell
& 'C:/Users/steen/anaconda3/python.exe' 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/variants/head_v2/scripts/make_gallery.py'
```

This edit imports modelling helpers from the preserved standalone generator and uses the frozen `.blend` as its input, following the brief's explicit exception to the newer scripted-character kit. It does not migrate to charkit, modify Claude's marine, or add cloth, rigging, UV or game work. The only supplied external artwork is Erik's original turnaround, SHA-256 `b80b4b84eec0a7f64ae905458eafb2b9da6f30c22cb91374801c8e96dfc766b6`. The gallery reference details are labelled crops/enlargements; model previews are unretouched Blender renders. The asset is **Space Marine (Codex)** in `art/characters/space_marine_codex/`. The 2026-10-05 [relocation manifest](../../relocation_manifest.json) verifies that the namespace migration changed no Blender or render bytes. Commands above target the normal checkout after integration; replace its root when using a worktree. This implementation agent did not push or merge.
