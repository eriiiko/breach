# Ship interior texture library — 2026-10-19

Normal-map follow-up: [version 2 catalogue](normal-v2/README.md) and
[review report](normal-v2/REPORT.md). This version preserves smoothing precision
and includes moving-light views, repeat checks and Laigter comparisons.
Original maps below remain preserved for comparison.

Generated incrementally with the built-in imagegen tool for Erik's texture exploration.
29 colour surfaces, each accompanied by a generated relief source, processed
16-bit height map and mathematically derived normal map. Browse [the catalogue](CATALOG.md).
Original colour PNGs are preserved at native resolution: 1254 x 1254 for surfaces,
1024 x 1536 for the transparent Persian rug. Prompted 2048 targets were not the
tool's delivered resolution; these have not been upscaled and called 2K assets.
No existing assets or level references have been replaced.

These are source material sets, not finished edge16 wall strips or complete PBR
sets. Generated relief is an artistic inference, not measured geometry. Reference
editing preserves the layout approximately; pixel registration, repeat seams,
physical scale and the final look still need review in the game before production
use. Normal-vector validity does not certify the inferred geometry or alignment.

## Delivery order

1. Clean white living quarters: realistic, pixel art, hand-painted cyberpunk anime.
2. Engine rooms: stainless steel industrial floors and walls.
3. Wood living quarters: honey-oak boards and removable veneer wall panels.
4. Captain's lodge: walnut parquet, walnut/brass walls and a separate Persian rug.
5. Garden / terrarium: limestone paths and sage-green wall cladding.
6. Aquarium / swimming hall: ceramic mosaics and aqua wall tiles.
7. Cockpit: ribbed flooring and navy/teal wall panels, drawn anime style.
8. Warehouse: cargo deck floors and reinforced wall panels.
9. Kitchen: terracotta quarry floors and white washable ceramic walls.
10. Toilets / washrooms: white hexagonal floors and mint ceramic walls.
11. Crew sleeping quarters: blue-grey linoleum and oatmeal acoustic walls.

## Using these sources

The current canonical pipeline is `art/tilesets/<name>/tileset.toml` through
`tools/bake_level_art.py`. Floors need variant strips. Walls need 16 edge16 pieces
and normal strips; these source images alone do not meet that contract. Room looks
must remain separate from physical material IDs. The current baker chooses one
AIR floor family per tileset, so per-room floor selection needs a later design.

A Persian rug should remain a separate rectangular texture, shown on a thin plane
just above the floor (or a decal when supported), with collision disabled. A 3D
block is only useful if its thickness should be visible. This library supplies
artwork; rug rendering is not implemented here.

## Maps and regeneration

- `<name>.png`: colour source (sRGB); the rug carries coverage alpha.
- `<name>_height_source.png`: raw generated relief reference (not calibrated depth).
- `<name>_height.png`: smoothed 16-bit scalar height; not physical displacement.
- `<name>_n.png`: tangent normal data, linear/non-colour, same dimensions as colour.
  Convention matches `tools/make_tileset.py`: normalize(-dh/dcol, -dh/drow, 1).
  Flat encodes (128, 128, 255). This is the 2D tile lighting convention, not a
  promise that green-channel orientation will suit every future 3D mesh UV layout.
  Rug normal coverage is copied exactly from the original colour PNG.
- `*.prompt.txt`: exact built-in imagegen prompts. The first realistic pair's
  colour prompts are in `prompts.md`; the rug's corrected relief prompt is the
  `height-revision` sidecar. The pixel folder is pixel-inspired: generated soft
  shading means it is not strict 128px pixel art.

Relief strengths (full normalized height range in output pixels): clean realistic
floor/wall 5/7; pixel 4/5; anime 4/6; industrial realistic 6/8 and anime 5/7;
oak 2/3; captain parquet/wall/rug 2/5/0.8; garden 4/5; aquarium 3/3;
cockpit 4/6; cargo 5/8; kitchen 3/3; washrooms 3/3; crew sleeping 1/4.
All use a 0.8px Gaussian height smoothing pass. Tile maps use wrapped smoothing
and gradients; the standalone rug does not wrap. This alone cannot repair a
nonmatching colour edge or guarantee seamless repetition.

The conversion extends the existing `tools/depth_to_normal.py`, reusing the
existing `make_tileset` normal kit. The authored-height path needs only numpy/PIL,
not torch, transformers or the isolated trial environment. It lives on branch
`art-texture-library-2026-10-19`, in `.claude/worktrees/texture-library`; the main
checkout's converter has not been changed. Example, from that worktree:

```powershell
& 'C:/Users/steen/anaconda3/python.exe' tools/depth_to_normal.py <height_source.png> <height.png> <normal.png> --height-input --strength 5 --blur 0.8 --tileable
```

For a rug omit `--tileable` and use `--alpha-from <rug.png>`. Strengths are artistic
starting values; they should be tuned with a moving light in the game. All styles
remain exploratory, not a selected game-wide direction.

## Validation and scope

Four focused converter property tests pass (flat, orientation/unit length,
periodic translation, alpha coverage). Broad `pytest tests -q -k
depth_to_normal_height` collection was attempted, but the fresh worktree has no
compiled `breach_physics` module and 101 unrelated collection errors; the focused
file passes independently. No gameplay, sim state, material IDs, renderer or level
has been changed. `validation.json` records per-asset dimensions and normal checks.
