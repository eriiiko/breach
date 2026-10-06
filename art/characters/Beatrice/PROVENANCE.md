# Beatrice — scripted Blender model (Claude), stages 1 and 2

2026-10-04. Tracker: eriiiko/breach #33. Branch `33-beatrice`. Offline asset files only. This stage:
body, bodysuit, hands and boots; the head is a bald placeholder (collection `Head`, objects
`Beatrice_Head`, `_Eye_L/R`, `_Ear`) for the later face-and-hair stage. No rig, no game file yet.

## Source and scope

- References, supplied by Erik (1024 x 1536 each); where they come from is not recorded here:
  - `concept.jfif` — the original artwork, three-quarter front, walking, cropped at the crotch. SHA-256
    `39770a90e118be1589fdfcbce4b691fbacacddf00abfc47a96129da35dcd25d4`.
  - `without_hair.png` — front view in the modelling pose (redrawn from the concept by an image model). SHA-256
    `e50b669f5fe177c219896d8c65b06947ce368cc2a0336ad483e7a04884836dc2`.
  - `with_hair.png` — the same with hair (for the next stage). SHA-256
    `7be681b2d6861440a265cea25e070b03d048be6a4af5ca35623a78c8a829b40c`.
  - `without_hair_side.png` — side view, same pose (supplied after stage 1). SHA-256
    `32ffb098b3f380c3a28a014312202c8792032202ce024d70b03aeace904cad78`.
  - `without_hair_back.png` — back view, same pose (supplied after stage 1). SHA-256
    `78c62cfe49669e59a78c8e63b973ad6281e755f8ad7409ef9edac7c6130cb98e`.
- Erik's steer: heights, limbs, shoulders, arms, torso above the waist, knees and below from the front
  view; hips, seat and thighs from the concept (the front view's are too wide); the front view's
  full-length suit. 1.68 m to the top of the head, boots on. Stage 2 (Erik): depths from the side view with the seat
  and thigh taken in (a moderate seat), a LOW block heel of about 3 cm, the back from the back view.
- No third-party geometry, textures or code.

## Reproduce

Blender 4.5, bundled Python. From `art/characters/`:

    bash charkit/run.sh Beatrice --sheet --turn --beauty all --save     # about 2.6 minutes
    bash charkit/run.sh Beatrice --sheet --shape sheet_hips --save      # the front view's hips, about 15 s
    bash charkit/run.sh Beatrice --variant crimson --beauty hero         # about 20 s
    bash charkit/run.sh Beatrice --compare                               # shape_vs_concept.jpg, about 1 min
    bash charkit/run.sh Beatrice --beauty shoulder,flank,flank_back,knee,boot,hand   # close-ups, about 50 s
    bash charkit/run.sh Beatrice --draft --sheet                         # about 15 s, the working loop

Outputs: `previews/` (tracked), `source/Beatrice.blend` (gitignored), `source/*mesh_stats.json`,
`source/*sheet_scores.json` (tracked).

## How it is built

- `scripts/beatrice.py` — the tables: `PALETTE`, `GLOSS`, `VARIANTS`, `LEGS` / `HIPS` / `TORSO` (the trunk
  rows), `SHAPES` (row overrides: `sheet_hips`), `CONCEPT` (the measured ratios), `SLEEVE`, `dims()`,
  `HEAD`, `BEAUTY`, `DEV_VIEWS`, `BANDS`. `scripts/sheet_ref.py` — the front view's pixel frame and the
  concept's placement. `scripts/build.py` — three lines over `charkit/suitbuild.py`.
- Shared kit, new: `charkit/bodysuit.py` (the skin-tight figure on the workers' `workwear.Figure`
  construction: half-body trunk loft welded at the mid-plane, set-in sleeve with the armhole envelope,
  plus body forms, signed `pipe` seams, mesh panels, stand collar, zip, cuffs, knee pads, heeled ankle
  boots) and `charkit/suitbuild.py` (its driver: band IoU, proportion ratios, shape variants, the
  concept comparison). Additive changes: `garment.panel` (a panel of any curved outline, CDT-triangulated),
  `wearmat.mat_gloss` / `mat_mesh`, `sheetfit.Sheet(fill_holes=, floor=)` (defaults unchanged).
- Materials: `beatrice_suit`, `beatrice_mesh`, `beatrice_skin`, `beatrice_eye`, `beatrice_boot`,
  `beatrice_sole`, `beatrice_metal`.

## Measured (stage 2)

- Concept ratios against the default build: hip/shoulder 0.911 → 0.914 (+0.3 %), hip/waist 1.794 → 1.817 (+1.3 %),
  thigh/hip 0.482 → 0.463 (−3.9 %).
- Silhouette IoU, default build, bands above the waist / waist–knee / knees down: front 0.947 / 0.870 / 0.900 (whole 0.901);
  side 0.918 / 0.917 / 0.869 (0.904); back 0.913 / 0.837 / 0.832 (0.859). `sheet_hips`: front 0.947 / 0.947 / 0.900,
  side 0.918 / 0.959 / 0.869, back 0.913 / 0.886 / 0.832.
- The front and back drawings disagree on where the lower legs stand (1.8 cm apart at the ankle); the leg centres split
  the difference (front knees-down 0.937 → 0.900, back 0.783 → 0.832). All three draw a high heel; the build wears the
  3 cm block heel.
- `source/mesh_stats.json`: 62 objects, 405,437 source / 1,129,616 evaluated triangles; all 62 closed, none open.

## Stage 2 construction (kit, additive)

- `suitbuild`: further views (`sheet_ref.VIEWS`) scored per band with `<view>_vs_reference.png`; the hips comparison
  (`tables.COMPARE`, `COMPARE_BAND`) in matte grey, front and side, outlines and labels drawn in Blender.
- `bodysuit`: back-view feature points (`"bx"`); seams as piping cords (`seam_style="tube"`); mesh panels set in the suit
  (`panel_rim="seam"`); toothed front and back zips with flat pulls (`zip.teeth`); a kneecap as a body form with a seam
  loop (`knee_pad=None`); the boot shaft's top feathered into the leg (`boot.feather`); cuff height/lift from the table.
- `garment.panel(surface_uv=)`, `wearmat.mat_mesh(surface=, coat=)`: the net laid in the panel's own coordinates (the
  world-plane net cut curved panels in shimmering contour rings).
- `parts.hand(thumb_dir=, fan=, knuckle=, finger_len=)`, passed through `workwear.build_hands`.

## Stage 3: second correction round (2026-10-04, answers `previews/review_stage2.md`)

Kit, all additive (every other character builds as before):

- `charkit/implicit.py` (new): signed-distance primitives (tapered superellipse capsules, ellipsoids,
  chains), smooth union/intersection, surface nets meshing with Newton projection onto the surface,
  Taubin smoothing.
- `parts.bare_hand` (new; `parts.hand` untouched): a bare hand as ONE closed implicit surface — palm
  rows wrist→knuckles, thenar / hypothenar / heel pads, a palm hollow, a two-segment thumb from the
  palm's base, four fingers of three tapering phalanges with knuckle waists and rounded tips, curl and
  splay; its rows inside the sleeve follow the forearm. `BARE_HAND` holds her slender hand.
- `bodysuit`: `build_bare_hands`; the shoulder as one surface (`garment["shoulder"]`): the smooth union
  of body and sleeve lofts (fillet k growing from the armpit to the shoulder top), meshed on its own and
  bounded by a raglan cut on seam lines (`Suit.in_torso_region`, `body_onto_cut`) and by the deltoid
  seam on the arm (`sleeve_to_join`, `Suit.t_join`); `sleeve_back` depths (the elbow's point);
  `build_boots_implicit` (`boot["implicit"]`: shaft = the leg eased out, heel cup, foot chain, flat
  underside, full-length sole, block heel); `Suit.teardrop` panels; per-panel material and the
  `mesh_thigh` net (palette `MESH_THIGH`); zip tooth size and pull scale; layered knee pads.
- `suitbuild.render_aimed`: close-ups aimed at any point (`DEV_VIEWS` entries given as dicts).

Her tables: the refitted `SLEEVE` + `SLEEVE_BACK`; the S-curve in `TORSO` (waist / waist_top / ribs /
chest_low / chest / chest_top / shoulder depths; hip, seat and thigh rows unchanged); lower-leg rows
1 mm out and the instep row slimmed (inside the boot); the raglan cut and arm seam (`RAGLAN_*`,
`ARM_SEAM`); teardrop inserts and the new thigh panel; joined-up seams; layered knee pads; a fine zip,
a darker `METAL`; thin cuffs; the implicit boot table; feet turned out 12°.

Measured (stage 3): silhouette IoU above the waist / waist�knee / knees down: front 0.946 / 0.871 / 0.908;
side 0.918 / 0.915 / 0.856; back 0.911 / 0.848 / 0.813 (back knees-down fell from 0.832: the lower legs sit 1 mm
further out for the front view). `source/mesh_stats.json`: 60 objects, 932,760 source / 2,322,320 evaluated
triangles, all closed.

## Stage 1 measures (superseded)

- Concept (by hand, far-side half-widths from the midline, concept px): shoulder 305, waist 155, hip at
  the crotch 278, far thigh at the crotch 268 → hip/shoulder 0.911, hip/waist 1.794, thigh/hip 0.482.
- Front view (measured by the driver): hip/shoulder 0.991, hip/waist 2.044, thigh/hip 0.461.
- Default build: hip/shoulder 0.909 (−0.2 %), hip/waist 1.840 (+2.6 %), thigh/hip 0.463 (−3.9 %).
- Silhouette IoU against the front view, default: whole 0.913; above the waist 0.963, waist–knee 0.873,
  knees down 0.924. `sheet_hips`: 0.948; 0.963 / 0.949 / 0.924.
- `source/mesh_stats.json`: 34 objects, 372,396 source / 1,076,652 evaluated triangles; all 34 closed and
  facing outward.

## Not done, and known weaknesses

The face is the head function's default (doll-like), for the next stage. See the stage-2 report for what still
reads weak (hands, the boot's heel counter, the shoulder cap seam, the back drawing's leg stance).

## B1e: the pelvis, second round (2026-10-05, answers `previews/review_stage4.md`)

- Reference added: `good-body-silhouette.jfif` (683 x 1024, supplied by Erik 2026-10-05; a front view, head to upper
  thighs, the earlier high-cut suit; "a good body shape, the same as the original image"). SHA-256
  `f1fd420732ad1fa18dab435340409632515b8d52a74a266510e97b93b9a8de43`. Body SHAPE only; nothing of its suit is used.
  Measured by hand (half-widths, px, between arm and body where the backdrop shows): deltoids 213.5 (row 480), narrowest
  waist 128 (row 760), widest 198.5 just above the crotch (rows 980-1000): hip/waist 1.55 (CONCEPT 1.794), hip/shoulder
  0.93 (0.911). Its waist-to-hip gain at 0.2 / 0.4 / 0.6 / 0.8 of the height: 0.30 / 0.56 / 0.78 / 0.93 (stage 4: 0.12 /
  0.36 / 0.59 / 0.79). Its frame: `scripts/sheet_ref.py::SILHOUETTE`.
- Kit (`bodysuit.py` and `hipsview.py` are this character's alone; no shared kit file touched):
  - `bodysuit.MidUnion` (`dims["midplane"]`): the pelvis as ONE surface across the mid-plane, the smooth union
    `smin(F(P), F(mirror P), k)` of the half-body (with its forms, continued past the plane) and its mirror image,
    k = 2 w sigma (`reach` w; sigma = the half-body's |n_x| where it crosses the plane, 1 under the crotch). On the loft's
    own grid: each vertex pushed horizontally within its level row onto the union, the swallowed run welded onto the
    union's own crossing of the plane, the stretch beside each weld point resampled (`_spread`), rows 0.1 mm apart at
    the arch's tip (`refine`), the tip relaxed and re-projected (`relax`), coincident weld vertices merged (`_clean`).
    Replaces `weld_rows` + `midline_fillet` from 0.80 to 1.125 m (both kept, used by the shape `stage4`).
    `Suit.forms(signed=)`, `Suit.body_lift`, `build_suit(body_only=)`.
  - `hipsview`: `crotch.jpg`, `shape_vs_silhouette.jpg`; `hips_before_after.jpg` now the before beside now (matte, four
    views); the measurements add the half-gap every 5 mm, `midplane_normals` and `mesh_check` to
    `source/<prefix>outline.json`.
- Tables: `PELVIS` (inner, outer, front, back), `SEAT` (broader, lower), `MIDPLANE`; `PELVIS4` / `SEAT4` / `MIDLINE` and the
  shape `stage4` (= `HIPS_BEFORE`); the slimmer / fuller steps' seats follow the new SEAT.
- Measured (full build): half-gap 6.0-8.0 mm from 0.83 to 0.86 m, 5.5 at 0.87, closed at 0.883 (stage 4: 21.0 / 14.8 /
  11.9, closed at 0.887); thigh/hip 0.479 (0.446; concept 0.482); hip/waist 1.795, hip/shoulder 0.859, hip width and
  waist unchanged. Suit_Body: 0 degenerate faces (stage 4: 520), 0 non-manifold edges; the other suit objects' counts
  are unchanged from stage 4 (cuffs, zip tapes, shoulder patch: pre-existing, outside the pelvis). Outline turning
  0.68-1.12 m, deg/cm, now (stage 4): front outer 3.31 at the waist, 1.8 below 1.10 (2.63); side front 2.76 (3.25);
  side back 4.11 at the waist, 3.73 at the seat (4.45); three-quarter front 2.20 (4.58); three-quarter back 3.91 at
  the waist, 3.54 at 0.888 (4.92). In step 1 the front and three-quarter outlines outside 0.78-0.93 m moved under
  1 mm (0.76 mm worst). Sheet IoU, hip-knee band, front / side / back: 0.853 / 0.907 / 0.833 (0.852 / 0.897 / 0.833).
- `source/mesh_stats.json`: 60 objects, 985,629 source / 2,534,292 evaluated triangles, all closed.
- Reproduce: `bash charkit/run.sh Beatrice --sheet --turn --beauty all --save` (about 6.5 min), `--hips` (about 10 min:
  stage 4 and the default, the crotch close-ups), `--compare`.

## B1f: the pelvis, third round (2026-10-06, answers `previews/review_stage5.md`)

No reference added. Kit: only Beatrice's own `bodysuit.py` and `hipsview.py` changed; no shared kit file touched.

- **The thighs meet (review A1, A2).** The inner profile (`beatrice.py::_gap`, `GAP`) is no longer a slot: the half-bodies
  touch the mid-plane tangentially at 0.866 (g = 0, g' = 0) and overlap above it; below it the half-gap is the cubic
  g = a dz^2 + b dz^3 (dz = 0.866 - z), convex, meeting B1e's leg at 0.78 in value and slope (below 0.78 the leg is B1e's);
  above it a quartic with the same curvature at the contact into B1e's overlap at 0.92 (C2 through the contact). The
  union's reach (`MIDPLANE`) is broad in the contact zone (8-12 mm: the groove where the thighs meet is rounded, not a
  pinch that shows as a bright line in gloss), and `bodysuit.MidUnion.k` takes a `core` cap, k <= k0 + c dy^2 at the
  sections' innermost depth (k0 3.2 mm, c 12 /m, full below 0.851, gone by 0.885), so the bridge still ends in a
  near-tangent cusp at the tip. Fine rows 0.836-0.906 and the relax box 0.838-0.925 (the back weld zipper).
  Measured (full build): first light at 0.851 m (B1e 0.880): 42 mm under the table's crotch landmark (0.893), 32 mm under
  the top of B1e's slot; half-gap 0.8 mm at 0.850, 1.7 at 0.845, 3.7 at 0.835, 7.9 at 0.820, 15.3 at 0.800, as B1e from
  0.80 down (full table, B1e beside it, in `source/outline.json` `half_gap` / `before.b1e.half_gap`); the inner outline's
  second derivative (a quadratic fitted over 6 mm every 1 mm, `hipsview.lens_measure`) is negative only over the bridge's
  last 6 mm (the tip's own arch, its half-gap under 2 mm) and positive from 0.845 down to 0.771, then negative into the knee
  (B1e's leg, unchanged). Suit_Body: 0 degenerate faces, 0 non-manifold edges, every object closed (the cuffs', zip tapes'
  and shoulder patch's counts are B1e's). In step 1 the front and three-quarter outlines outside 0.78-0.93 m moved at most
  0.01 mm; the quarter outlines' worst turn there 2.97 deg/cm.
- **The side (review: "it barely moved").** The back outline was designed by its slope angle (at most 2.9 deg/cm by design,
  integrated) and turned into profile controls; the seat form (`SEAT`) is taller, narrower and falls off sooner below its
  centre (`Suit.forms` takes an optional 7th field). Largest separation from B1e: back +7.4 mm at 0.984, +6.3 at 0.94;
  front +5.3 mm at 0.814 (`side_outlines.jpg`). Thigh-front swell from the 0.90-0.62 chord: 17.3 mm (B1e 14.1, stage 4
  13.1); on the thigh alone (0.88-0.70 chord) 13.3 (9.6). Seat round from the lumbar (1.144) - 0.70 chord: 49.2 mm (42.8);
  the seat about half the way to the side drawing ((98.9 - 92.5) / (104.5 - 92.5)); the lumbar hollow 0.9 mm deeper at 1.08.
  Side turning 0.68-1.105: front 2.89, back 3.78 at the seat's crown (B1e 3.73): not under 3.
- **The front.** `_flare` moves B1e's outer controls between the narrowest waist (1.138) and the hip line (0.893) a `step` of
  the way to the curve with the owner's reference shares. Measured (`hipsview.flare_shares`, on the outline, the same way for
  every shape): stage 4 0.135 / 0.389 / 0.636 / 0.845; B1e 0.153 / 0.430 / 0.683 / 0.875; the full step (shape
  `flare_full`) 0.283 / 0.548 / 0.773 / 0.927; the default, a half step, 0.206 / 0.478 / 0.722 / 0.899. The full step makes
  the waist a corner (outer turning at the waist 17.4 deg/cm, B1e 6.9; a kink in the highlight in gloss) and the hip
  straight-sided, so the default stops at half (waist joint 11.3). Waist 90.0 mm, widest 162.1 (B1e 162.3), hip at 0.893
  157.8 (157.7). The outer thigh tapers 1.5-1.8 mm toward the knee (`TAPER`). The bands on the thigh in the matte
  three-quarter front were curvature concentrations (quarter outline turning 2.9 at 0.66, 2.0 at 0.77, 2.2 at 0.88): the
  generated rows now start at the knee (`PELVIS["z"]` from 0.605; their junction with the sparse leg rows was the 0.66
  band), and the new front and lens profiles spread the other two (now 0.74 at 0.66-0.68, 1.83 worst in the band);
  `bands_contrast.jpg`.
- **The zip tab (review A4)** was the centre-back seam's piping: at 1.07 the union's lift of a point on the plane ran 3 cm
  out of the back, and below it the cord left the centre line across the left buttock. `bodysuit.midline_cord` lays a seam
  drawn at phi +-90 inside the midplane region on the union's own crossing of the plane. The zips are unchanged.
- `Suit_Body`'s hidden inner shell is solidified without even offset (an even-offset spike at a raglan-cut sliver, x 0.118,
  z 1.32, once the pelvis rows moved).
- Shapes added: `b1e`, `flare_3q`, `flare_full`; `HIPS_BEFORE` = stage 4, B1e, flare_full. Pictures added: `lens.jpg`,
  `side_outlines.jpg`, `bands_contrast.jpg` (made from `hips.jpg` and B1e's, commit 177291a, by a scratch script: contrast
  stretched and high-passed).
- Sheet IoU (above the waist / waist-knee / knees down): front 0.946 / 0.857 / 0.908, side 0.918 / 0.921 / 0.856, back
  0.912 / 0.841 / 0.812 (B1e waist-knee 0.853 / 0.907 / 0.833). `source/mesh_stats.json`: 60 objects, 1,018,312 source /
  2,664,932 evaluated triangles, all closed (the flipped-normal list is B1e's).
- Reproduce: `--sheet --turn --beauty all --save` (6.5 min), `--hips` (20 min: four builds), `--compare` (13 min), and
  `--beauty shoulder,flank,flank_back,knee,boot,hand`.
