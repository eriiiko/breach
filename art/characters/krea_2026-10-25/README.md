# Krea characters, 2026-10-25 batch — working folder

Nine characters Erik made in Krea (Meshy V7, ~31k tris, ~1.9 m), and the
char_04 work done on 2026-11-02/03. Working files, NOT game-ready (not on the
SOMA skeleton, 2k textures, ~16 MB each).

## Where the big files are

Git holds scripts, sheets and the source list only (storage ruling 2026-11-02:
raw and working GLBs stay off git). The GLBs are on Google Drive:
`G:\Min enhet\Breach\krea_transfer_2026-11-03\`

- `characters_raw/char_01..09.glb` — also re-downloadable: `sh fetch_sources.sh`
- `char_04/` — recolours (`char_04_yellowwhite_blackgloves.glb` = Erik's pick:
  shell 0.95,0.91,0.62, Ray-Ban green visor, black gloves), `char_04_headless.glb`
  (head carved out inside the kept collar), `char_04_beak_{red,ivory}.glb` (headless
  body + seated helmet, for viewing)
- `helmet_beak/` — raw Tripo helmets (`helmet_text_tripo.glb` = round goggles +
  beak, Erik's first pick; `helmet_b_tripo_img.glb` = crest + goggle band), and the
  40k-tri game meshes `helmet_beak_{orig,ivory,red}.glb` seated in char_04's frame
  (lenses recoloured to the visor green; `orig` keeps the turquoise)

## Rebuild

Blender 4.5 headless (`blender -b -P script -- args`); each script's docstring
gives its arguments.

1. `char_04_recolor/`: `extract_tex.py` -> `visor_faces.py` (head_tris.json) ->
   `recolor.py` -> `apply_render.py`
2. `char_04_helmet/`: `helmet_prep.py` (weld + decimate the raw helmet, save its
   base colour) -> `recolor_helmet.py` (paint + lenses) -> `fit_helmet.py` (carve
   the head ellipsoid, seat the helmet: `--seat -0.085,0.815,0.23`)

## Open

- Bind to SOMA (`game-asset-pipeline` skill: `blender/retarget_soma.py` in
  local-models, Work Desktop). The helmet is its own asset, to ride the head bone.
- Hunyuan licence: none of these assets used Hunyuan (Meshy + Tripo only).
