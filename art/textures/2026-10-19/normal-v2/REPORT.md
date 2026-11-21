# Normal-map review and delivery — 2026-10-19

The selected method is the existing generated relief reference, smoothed as a
scalar field, then converted through `tools/depth_to_normal.py` and the
`tools/make_tileset.py` normal kit. Version 2 supplies all 29 surfaces. Colour
artwork, source relief, and the original maps are preserved. These are source
material candidates for game integration, not finished edge16 tilesets.

## What changed

The original converter blurred through a uint8 image even for uint16 input.
Consequently the original height outputs had only 104–251 distinct levels.
The converter now applies a separable Gaussian directly in floating point,
then quantizes once to the output 16-bit height. Version 2 has 20,352–54,404
distinct levels, preserving smoothing intermediates and shallow relief.
This is numerical precision, not extra measured geometric information: the
generated source images are still artistic estimates of surface elevation.

The documented material strengths are retained: shallow wood and linoleum,
moderate ceramic grout, stronger industrial panel joints. There is no evidence
yet to justify changing their scale globally. No colour texture was regenerated.
No new AI generation was used for v2: all source prompts remain beside the
original relief references.

## Visual review

The five sheets cover every colour/height/normal/neutral-lit material:
[1](review/sheet-01.jpg), [2](review/sheet-02.jpg),
[3](review/sheet-03.jpg), [4](review/sheet-04.jpg),
[5](review/sheet-05.jpg).

Each catalogue row also links a rotating-light animation and a 2x2 repeat.
Neutral lighting exposes inferred relief without the colour image disguising
it. The colour-lit view is a second check for obvious misregistration. The light
rotates in the canonical image-row-down normal frame at a grazing elevation.
This simple Lambert diagnostic deliberately does not duplicate Breach's HDR
renderer, sweep, or tone-map. Game look shots remain the integration gate.

Visual assessment of the source families:

| Family | Assessment and remaining concern |
|---|---|
| Clean quarters, three styles | Broad panels remain flat; joints and fittings read. Realistic floor has much more fine woven relief than the other floors; review at intended screen size. Pixel style remains pixel-inspired rather than strict pixel art. |
| Engine room, two styles | Generated relief separates paint from structure better than the tested direct converter defaults. Diamond plate still contains directional-looking profiles in the inferred height; raised-diamond shape is approximate. |
| Oak | Conservative joints and fine grain suit finished boards better than a deeply embossed grain. Exact seam registration still needs full-resolution integration review. |
| Captain's lodge | Parquet joints and wall trim read; rug stays almost flat and keeps the source alpha. Pattern colours should not become carved embroidery. |
| Garden | Limestone retains pores and joints; sage wall removes most colour variation from relief. Pore depths are artistic. |
| Aquarium/pool | Colour changes between adjacent tiles do not become large height changes. Repetition of the source mosaic still shows border/pattern discontinuities. |
| Cockpit | Ribs, joints and vents read while painted marks are largely suppressed. Some microrelief can alias when reduced to game scale. |
| Warehouse | Cargo floor keeps tread and panel joints; wall panels have stronger borders. Avoid exaggerating this again through a global normal-strength setting. |
| Kitchen | Quarry and white ceramic are shallow, with recessed grout. Source grout spacing is not perfectly procedural or guaranteed periodic. |
| Washroom | Dark accent hexagons stay at tile elevation instead of becoming deep pits. The repeated source hex pattern has visible boundary discontinuities; it is not certified seamless. |
| Crew sleeping | Linoleum stays close to flat; acoustic panels retain fine structure. Review fabric noise at the game's pixel density. |

## Laigter comparison

The installed executable is
`C:/Users/steen/projects/laigter/dist_win/laigter_app/laigter-windows/laigter.exe`.
Three baseline conversions were run headlessly with installed defaults, with
outputs isolated under `../normal-review/laigter-baseline/`. Commands and basic
results are recorded in [laigter-baseline.json](../normal-review/laigter-baseline.json).

- [Metal plate comparison](../normal-review/02-engine-room--realistic--floor--comparison.gif)
- [Oak comparison](../normal-review/03-wood-quarters--realistic--floor-oak--comparison.gif)
- [Hex ceramic comparison](../normal-review/10-washrooms--realistic--floor-hex--comparison.gif)

The default baseline introduces broad plate bulging and treats more of the
colour/lighting variation as relief. The generated-height route is selected
for this delivery because it better preserves flat surfaces and distinguishes
paint from elevation. This does not establish the best result achievable with
a tuned Laigter preset; no preset tuning or green-channel flip was applied.
Orientation differences must be judged separately from geometric artifacts.

## Validation and use

All 29 normals have matching colour dimensions, positive decoded Z, and maximum
normal length error below 0.007 after RGB8 encoding. All height files are uint16;
surfaces retain 1254 x 1254 and the rug 1024 x 1536. Rug alpha matches the source
exactly and fully transparent pixels carry flat normals. See
[manifest.json](manifest.json) for per-surface results, paths and strengths.

The converter's five focused property tests pass. The new regression test
protects a shallow uint16 ramp that the old uint8 blur erases. Existing gates
cover flat relief, slope orientation, periodic translation, and alpha coverage.
No sim, renderer, level, or config code changed; the broad physics suite was not
rerun for this asset/converter change.

Normals are linear/non-colour data in the existing tile-lighting convention:
`normalize(-dh/dcol, -dh/drow, 1)`. Flat is `(128,128,255)`. Do not infer a universal
3D OpenGL/DirectX green-channel convention from this row-down tile frame.
Heights are normalized artistic elevation, not calibrated displacement in metres.

The original source images do not guarantee periodic colour, geometry, or
pixel-perfect registration. Wrapping smoothing/gradients makes the operation
periodic; it does not repair those source defects. Integration should use the
existing tileset/baker path and named look shots, with Erik's visual gate before
shipping. That requires selecting floor variants and constructing wall edge16
pieces, a separate task from supplying these source maps.

## Reproduction and systems

From this branch's worktree, using this machine's breach Python:

```powershell
& 'C:/Users/steen/anaconda3/python.exe' art/textures/2026-10-19/build_normal_v2.py
& 'C:/Users/steen/anaconda3/python.exe' art/textures/2026-10-19/review_maps.py --v2
& 'C:/Users/steen/anaconda3/python.exe' art/textures/2026-10-19/compare_maps.py
& 'C:/Users/steen/anaconda3/python.exe' -m pytest tests/test_depth_to_normal_height.py -q
```

Canonical systems reused: `depth_to_normal.py`, `make_tileset`'s normal kit,
the original catalogue and relief sources. The local review scripts produce
offline asset diagnostics, not a second game screenshot harness. No new runtime
system is introduced. Work is isolated on `art-normal-review-2026-10-19`, based
on the existing texture-library branch because that branch owns these assets
and the authored-height converter. Nothing was merged into an integration line.
