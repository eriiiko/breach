# Space Marine (Codex) - high-detail source

Approved original, completed 2026-10-04 and accepted by Erik on 2026-10-05. Open [the local review gallery](previews/index.html) for the supplied reference and seven actual model renders. Open [source/space_marine.blend](source/space_marine.blend) to inspect or edit the character. It opens in a material viewport with the studio helpers hidden and the visor selected. The complete character can be moved through its `SPACE MARINE / move the complete source` parent.

The [approved-v1 snapshot](versions/approved_v1/README.md) preserves this original. The separately reviewed [head-v2 gallery](variants/head_v2/previews/index.html) compares it with the additional helmet variant. Both are retained as Space Marine (Codex).

This pass retains the approved silhouette, equipment, materials and existing authored fold shapes. The elbow and knee liners now tuck beneath the ivory garment ends, the concealed glove binding sits inside its black cuff, and stitch ends enter the fabric. The separate cloth experiment has not been applied.

## Independently adjustable visor

- Collection: `03 / Visor glass selection`.
- Object: `VISOR GLASS / convex optical shield`.
- Material: `VISOR GLASS / olive optical coating`.
- Direct Principled controls: **Roughness 0.135**, **Coat Weight 0.55**, **Coat Roughness 0.07**. These are ordinary editable values, without drivers or a `game_gloss` override.
- Object Index and Material Index **1** identify only the glass. All other asset objects/materials use 0. Both index passes are enabled. Use Render Result or a multilayer EXR to retain those mask passes; the gallery PNGs are beauty images.

The helmet shell, retaining lip and black gasket remain separate geometry and materials. The glass material is used only by the glass object. An embedded Blender text block, `README / source and visor controls`, repeats the selection instructions.

The later `art/characters/charkit/gameready.py::gloss_signal`, inspected on the separate `fire-12` branch and not included in this asset package, reads Principled Roughness, Coat Weight and Coat Roughness and optionally accepts material `game_gloss` / `game_gloss_scale` overrides. This source uses the standard inputs so the later workflow can derive its gloss signal. That is compatibility with the documented input convention, **not a claim that an export, bake or engine test was performed**.

## Measured source and review limits

The reopened-file audit in [source/validation.json](source/validation.json) reports:

| Measurement | Value |
|---|---:|
| Character mesh/curve/text objects | 460 |
| Source mesh vertices | 446,474 |
| Source mesh polygons | 442,133 |
| Evaluated vertices, including curve detail | 626,789 |
| Evaluated triangles | **1,245,974** |
| Overall posed dimensions | 1.083 × 0.498 × 1.881 m |
| Ground level / forward / up | Feet Z=0 / −Y / +Z |

The high-detail geometry is intentionally retained. No decimation, game mesh, UV atlas, texture bake, rig, skin weights, LOD or runtime export was added. The `.blend` contains the packed reference, editable procedural materials, named review cameras and studio lighting. It has no live cloth modifier or external simulation-cache dependency. Layered armour and garment components overlap intentionally; this is not a single watertight mesh for printing. The audit records 234 evaluated triangles below 10⁻¹² m²; it is not a game-topology acceptance report.

The saved source was reopened in Blender 4.5.10, all seven delivered views were visually inspected, and all 21 gallery file links resolved. The render manifest's source SHA-256 matches the delivered `.blend`. The audit found no missing external image, nonfinite position, unexpected armature, live cloth modifier or reused visor mask ID. The parent reviewer accepted the named joint-contact cleanup; Erik approved the model and authorized integration on 2026-10-05.

The remaining visible weakness is the stylized, sometimes cord-like diagonal fabric creases and their sampling. This cleanup deliberately preserves them. The completed [bounded sleeve comparison](experiments/compressed_cloth/README.md) tested two compressed-cloth candidates. Both were rejected for rollout: neither clearly improved the complete padded sleeve. Their editable study sources and comparisons are retained separately.

## Reproduce or render a hand-edited source

These PowerShell commands use the normal checkout location after integration. In another checkout, replace `C:/Users/steen/projects/breach` with its root; scripts resolve asset paths from their own location. Blender 4.5.10 LTS was used. The authoring and rendering scripts require only Blender; the contact-sheet script additionally uses Pillow, available in the local Anaconda base environment.

**Render the existing saved `.blend` without rebuilding or overwriting it:**

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/source/space_marine.blend' --python 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/scripts/render_space_marine.py'
```

This renders front, side, back, hero, helmet close-up, elevated and lower-body detail views using Cycles, 160 samples and denoising. Most images are 1600 × 2000; the upper-body close-up is 1800 × 1600. To render selected views, append `-- --views hero,closeup`. Append `-- --draft` for smaller previews. The script verifies that the saved source hash is unchanged and writes [previews/render_manifest.json](previews/render_manifest.json).

**Rebuild from the generator — this replaces `source/space_marine.blend` and its source statistics:**

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/scripts/build_space_marine.py' -- --no-render
```

**Reopen and inspect the saved source:**

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/source/space_marine.blend' --python 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/scripts/inspect_space_marine.py'
```

**Recreate the labelled contact sheet from the rendered PNGs:**

```powershell
& 'C:/Users/steen/anaconda3/python.exe' 'C:/Users/steen/projects/breach/art/characters/space_marine_codex/scripts/make_contact_sheet.py'
```

## Provenance and scope

The authoritative image is Erik's supplied `art/characters/Space Marine Turnaround Sheet.png`, SHA-256 `b80b4b84eec0a7f64ae905458eafb2b9da6f30c22cb91374801c8e96dfc766b6`. It is packed inside the `.blend`; the gallery also links the original reference. The modelling functions, geometry and procedural materials were authored for this asset. Blender's built-in font supplies the small equipment labels. No downloaded model, texture, HDRI or AI-generated preview was used.

The `scripted-character` skill was read on the primary checkout, subsequently identified as `fire-12`, not `main`. That skill and charkit are absent from integration baseline `ef9d39c` and were not imported with this asset. The [existing brief](../space_marine_brief_2026-10-03.md) records the dated run learning and the explicit exception retaining this older standalone generator. Game preparation remains deferred to Erik's workflow. All-angle source review remains the criterion, with no commitment to a future game camera.

The earlier `art/characters/agent/` asset, the canonical skeleton, the renderer and the deterministic simulation are untouched. The 2026-10-05 namespace migration to `space_marine_codex` changes paths and text only. [relocation_manifest.json](relocation_manifest.json) records the original hashes and verifies that all 40 tracked Blender/render files remain byte-identical. The approved and study manifests retain historical hash maps alongside their current text hashes.
