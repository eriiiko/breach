---
paths:
  - "art/characters/**"
  - "assets/models/**"
---

# Characters

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. The
`scripted-character` skill is the procedure; new characters and props come from
the `game-asset-pipeline` skill.

## Character kit
- The scripted-Blender kit for the EXISTING scripted characters. NOT the default for new characters (Erik, 2026-11-02: the scripted Beatrice's quality was too low): new characters and props come from the generated-asset pipeline (Flux Kontext → TRELLIS.2; Beatrice #154).
- Character models carry NO props (Erik's rule): a bag or a weapon is its own asset.

## Game-ready step
- The ONLY way a scripted character reaches the game: one fused low-poly skin with a baked texture, bound to the game's ONE humanoid skeleton, with the clips converted to the character's own rest pose. Never a hand export, a second skeleton, or a mesh bent into the rig's T-pose. The skeleton is moving to Kimodo's SOMA (D1 LOCKED 2026-11-03, `docs/animation_milestones_design_2026-11-02.md`); the Quaternius clips are not ported.
- The bake writes the GLOSS mask into the albedo's alpha from the source materials' roughness on every path, and the albedo PNG is spliced into the exported .glb byte for byte (`art/characters/charkit/glbtex.py`), never left to the exporter's re-encode. A texture-only change is `--retexture` (mesh, skin and clips checked unchanged).

## Toon look
- Render-time only, converting an opened build: never saved into it, never the game's renderer. A new character's look is its own table on this module, never a copy (#33).

## Worker builder
- A coverall-clad human is a set of TABLES on this builder, never a copy of it. Colours live only in `PALETTE`; the `worker_*` material names are the contract with the game step; a colour variant is a re-bake onto the same mesh. The second member of any character family turns the first one's script into such a builder.
