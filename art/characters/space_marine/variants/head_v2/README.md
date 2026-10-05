# Space Marine (Codex): head v2

First candidate checkpoint, 2026-10-05. This is a separate full-character source with a helmet-only edit. The approved model remains available as a byte-identical [approved-v1 snapshot](../../versions/approved_v1/space_marine_codex_approved_v1.blend) and in its [original gallery](../../previews/index.html).

Candidate source: [space_marine_codex_head_v2.blend](source/space_marine_codex_head_v2.blend). The initial comparison is under `previews/review_1/`, with matched `approved_v1_` and `head_v2_` front, side, back and hero images plus the variant's full-figure hero. All are actual Blender renders, using the original lighting.

The compact shell has separate plates over a recessed substrate, a more projecting side-wrapped visor, a shaped gasket/retainer and curved temple carriers. The original collar, body and pose are retained. `source/preservation_fingerprints.json` records all non-head evaluated geometry, world transforms, material assignments and procedural material controls. The saved source was reopened and compared against these fingerprints in `source/validation.json`. All 18 frozen original files and the approved snapshot are hash-checked before and after rendering.

Initial implementation review: the side projection and temple construction are clearer, but the frontal glass still reads too oval, the chin frame remains thin, and the circular crown cap and rear equatorial seam may be too simple/bold. Parent review is pending before the one authorized focused correction. The candidate does not supersede the approved original.

Rebuild only this variant, then render a matched draft comparison and audit:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python 'C:/Users/steen/projects/breach/.claude/worktrees/33-agent-blockout/art/characters/space_marine/variants/head_v2/scripts/head_variant.py' -- --build --render both --draft --audit
```

To render existing saved sources without rebuilding or saving them, omit `--build`. Omit `--draft` for final 1600-pixel head views and a 1600 x 2000 full-figure hero. `--render head_v2` renders only the variant. `--audit` reopens and validates the saved variant.

This edit imports modelling helpers from the preserved standalone generator and uses the frozen `.blend` as its input. It does not rebuild the body, migrate to charkit, modify Claude's marine, or add cloth, rigging, UV or game work. The separate visor retains Object/Material mask ID 1 and ordinary Principled roughness/coat controls.
