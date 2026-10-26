# Kessler — provenance

A heavy EVA salvage marine. Named by Erik, 2026-10-26.

## Source

- `reference/kessler_sheet.png` (plus `SpaceSuitMarine_front.png`/`_back.png`): the T-pose
  concept sheet Erik generated (Krea), 2026-10-25.
- `source/kessler_source.glb`: an AI image-to-3D model generated from that sheet in Krea by Erik,
  downloaded 2026-10-26. It is one mesh (48,164 vertices, 31,016 faces), with one material and three
  2048 px textures: base colour, glTF metallic-roughness (`Image_1`, roughness in G) and a normal map.
  It has no rig. It is 1.71 m tall, stands in a T-pose with the arms ~10° below horizontal, and faces −Y in Blender.
  Its UV layout is the AI tools' scrap atlas, so the texture cannot be painted by hand in any useful way.
  An OBJ export of the same model was also downloaded; it is not kept, because the GLB holds the same data.
- Licence: AI-generated on Erik's Krea account. Commercial-use terms follow Erik's Krea plan
  (to be confirmed by Erik before any release).

## Look decisions (Erik, 2026-10-26)

- **The visor** is replaced. The model's own visor is lumpy, low-detail geometry, so a sphere is fitted to it
  and a new, smooth glass cap is cut to the helmet opening (`scripts/lookdev.py`, `visor_cap`).
  Its colour is a setting: deep green glass is the default (option 6 in `previews/lookdev/visor_options.png`).
  Erik: "satisfied with the visor"; the colour can change later.
- **The helmet lamps** (one on each side of the helmet) are selected by position and brightness (`lookdev.py`, `lamp`).
  Making them glow waits for the flashlight feature, which is not built yet. The selection is kept so that work starts from it.
