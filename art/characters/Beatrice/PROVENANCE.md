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

## B1g: the waist and the finish (2026-10-06, answers `previews/review_stage6.md` N1-N5, N7)

No reference added. Kit: only Beatrice's own `profiles.py` (added `angle_controls`) and `bodysuit.py` (`_onto_crossing`, the
zip stop) changed; no shared kit file touched. The side profile, the outer-thigh panel, boots, hands, knee pads, shoulders and
head are untouched. Shapes added: `b1f` (B1f's default, as `b1e` and `stage4` are kept); `flare_3q` / `flare_full` now
derive from B1f's pelvis explicitly. `HIPS_BEFORE` = stage 4, B1e, B1f.

- **The waist (N1).** B1f's narrowest waist was where two straight cones met (the rib cage coming down, the hip leaving),
  turning 11.3 deg/cm in a few mm. The outlines from the hip to under the bust are now authored by their ANGLE to the
  vertical (`profiles.angle_controls`: theta(z) a C2 cubic through the knots, the outline its integral; tables `WAIST`,
  `_waist`), the profile region running on from 1.135 to 1.265 through the old waist / waist_top / ribs / chest_low rows (no
  longer rows of their own; `_rows` takes the three TORSO rows above the region as its upper controls). The waist is one
  turn spread over about 10 cm. Measured on the built body (1.00-1.25 m; above 1.225 the bust form and the armpit, not part
  of this): front outer 11.33 -> 3.90 deg/cm (step 1 alone 3.23), side back 6.71 -> 3.40, side front (to 1.20) 3.82 -> 1.81,
  three-quarter front 8.11 -> 3.23, three-quarter back 8.56 -> 4.21 (at 1.176, the rib cage's turn). Narrowest 90.04 mm at 1.142 -> 90.00 at 1.136;
  the outline within 1 mm of it over 34-36 mm of height (B1f 18), within 2 mm over 50 (26). Ribs (1.22) 112.0 -> 112.0,
  chest (1.285) 128.0 -> 128.0; depths at the waist: front -104.0 -> -104.4, back 42.0 -> 42.0 (the lumbar's deepest point
  6 mm higher). The side view's front (`WAIST["front"]`, a third pass): at the centre front the glossy reflection's edge
  across the abdomen is where that outline's normal passes through the studio's horizon, and its sharpness goes with how fast
  the outline turns there, so its turn from the belly's crest into the rib cage is spread as far as the depths allow (at
  most 0.5 mm from B1f's): 2.7 -> about 1.6 deg/cm at the waist. By eye: the matte waist shows no line in any view; in the
  gloss the crisp edge of B1f is a soft gradient now (`hero.jpg` clearly, `torso.jpg` less so: light above, dark below, over
  2-3 cm instead of 1-1.5). Picture: `waist.jpg`.
- **The flare (N2).** The outer outline is authored by angle from above the knee (0.66): the outer thigh tapers 3 mm toward
  the knee (0.72) from its widest (0.81) instead of standing as a column; the hip is one convex round up to the steepest flank
  (27 deg at 1.06) and turns there into the waist's long concave turn. Widest 162.1 at 0.812 (B1f 162.1 at 0.82), hip line
  (0.893) 157.8 (157.7), waist 90.0. Flare shares at 0.2 / 0.4 / 0.6 / 0.8: 0.127 / 0.471 / 0.772 / 0.921 (B1f 0.206 / 0.478 / 0.722 / 0.899; the
  full step 0.283 / 0.548 / 0.773 / 0.927; three quarters of the way 0.264 / 0.530 / 0.760 / 0.920). The 0.6 and 0.8 shares
  reach the three-quarter target; the 0.2 and 0.4 shares can not while the waist turns at most 4 deg/cm: with zero slope at
  the narrowest point, 4 deg/cm allows about 8 mm of gain in the first 5 cm (share 0.13), and the reference's 0.30 there
  would need a turn of about 9 deg/cm -- the corner N1 removes. So the hip is rounder and fuller lower down, slimmer than B1f
  just under the waist (up to 7 mm at 1.08). Front turning over the pelvis band (0.66-1.05) at most 2.93 deg/cm (0.996); the hip pictures' title value (4.5 at 0.608) is the knee's join, B1f's 4.6.
  Pictures: `hips.jpg`, `hips_before_after.jpg`, `shape_vs_silhouette.jpg`, `shape_vs_concept.jpg`.
- **The crotch (N3, N4, N5).**
  - N3, the dark comma at the seam V's tip: the leotard line (and the seat's U behind) runs into the mid-plane inside the
    midplane union; the lift pushed the union-swallowed points TO the plane along their normal, up to 9 mm under the
    surface, so the cord's end dived into the suit. `bodysuit._onto_crossing` cuts the swallowed run at a cord's end back to
    one point on the union's own crossing of the plane: the cord and its mirror meet on the surface (`crotch.jpg` glossy).
  - N3, the thin bright line from the V's tip down to the gap: cause found, NOT fixed. The groove where the thighs meet is a
    round of 2-3 mm (faces meeting the plane at |n_x| 0.3-0.5 from 0.865 to 0.88), so a forward-facing strip about 1 mm wide
    catches the light. A broader fillet (front reach 15 and 20 mm from 0.865 to 0.895, the relax box 22 and 30 mm wide, the
    march's reach 40 mm) halves those |n_x| and in the gloss lets the belly's highlight run down into the contact as one
    wedge, with first light unchanged (0.851) -- but it draws horizontal stripes in the matte hollow above the cusp and a hard
    edge at cusp height in the three-quarter view (five builds); the reach stays B1f's.
  - N4: from 0.70 to 0.80 the inner outline is authored by angle (`GAP_LEG`, `_gap_leg`). A convex lens can not run straight
    into the knee's stance (its steepest slope, 0.47, is twice the knee's, 0.24), so the slope now eases back over the whole
    stretch: the concave part is ONE even taper from 0.705 to 0.78, at most -3.9 /m (0.751; B1f: an -8.7 /m hinge at 0.761; hipsview.lens_measure), the
    lens convex from 0.78 up (B1f 0.771). First light 0.851, the half-gap from 0.80 up unchanged. `inner_thigh.jpg`.
  - N5: the 0.80 band in the matte three-quarter front is gone (it went with N4's hinge); the finest crotch rows (0.1 mm,
    0.845-0.853) grade into the 0.6 mm rows over 6 mm instead of 2, which softens the cusp-height streak; the short streak
    beside the crotch is still faintly there in the high-pass (`bands_contrast.jpg`). N6 (the needle tip) left.
- **The zip's foot (N7)**: the stop at the foot of each zip (`zip["stop"]`) is a flat plate 6 x 4.5 x 1.1 mm, 0.4 mm of it
  sunk, on the figure's own surface; before, a 2.5 mm block floating 0.8 mm off the loft (3.3 mm proud). Nothing else on the
  zips changed.
- Sheet IoU (above the waist / waist-knee / knees down): front 0.947 / 0.852 / 0.908, side 0.918 / 0.920 / 0.856, back 0.914 /
  0.837 / 0.812 (B1f 0.946 / 0.857 / 0.908, 0.918 / 0.921 / 0.856, 0.912 / 0.841 / 0.812). `source/mesh_stats.json`: 60
  objects, 1,031,688 source / 2,718,684 evaluated triangles. Mesh check: Suit_Body 0 degenerate faces, 0 non-manifold edges;
  the cuffs' and zip tapes' degenerate counts are B1f's; Suit_Shoulder's non-manifold edges 22 -> 32 (its raglan-cut join with
  the body rows, which are now 5 mm apart up to 1.265 there).
- `crimson_hero.jpg` was not re-rendered (the variant run is not part of this set): it shows B1f's body.
- Reproduce: `--sheet --turn --beauty all --save` (6.6 min), `--beauty shoulder,flank,flank_back,knee,boot,hand` (4.8 min), `--hips`
  (19.8 min: stage 4, B1e, B1f and the default), `--compare` (13.0 min). `waist.jpg`, `inner_thigh.jpg` and `bands_contrast.jpg` are made
  from those by scratch scripts (PIL / matplotlib; the waist's matte crops from body-only builds of `b1f` and the default).

## B1h: the legs below the knee (2026-10-08, answers `previews/review_stage7.md` B1, B5, B8)

No reference added. Kit: only Beatrice's own `bodysuit.py` changed (new: `FittedBoot`, `build_boots_fitted`, `smin3` / `smax3`,
`build_knee_pads`); no shared kit file touched, so no other character was rebuilt. Nothing above the knee pads moved: every
evaluated vertex of every object above z 0.652 (834,604 of them) lies within 8.3e-8 m (float32 rounding) of a B1g vertex and
the other way round (`Beatrice_b1g_9b6d9df.blend` against this build). To keep it so, the pelvis profiles' lower controls (the
three leg rows under the knee, `profiles.edge_rows(below=)`) stay B1g's (`LEGS_B1G`), and the suit's leg still ends at 0.10
inside the boot (ending it at 0.15 moved 20 suit vertices at the raglan cut by 0.009 mm: the body build is order-sensitive).

- **The boot (B1).** `boot["fitted"]` replaces `boot["implicit"]` (kept, unused). The causes: the shaft was the leg loft
  eased out 2.2-4.8 mm over a hidden ankle row whose back bulged 6 mm behind the visible ankle (a tube standing off the
  Achilles); its top was a flat cut (a rim) with a V dip; the foot a chain of eight tapered superellipse capsules unioned at
  every node, each node's rounded end a ring of bump and dent (the vamp's blotches: the reflection lines in
  `boots_before_after.jpg` show the rings), smoothed by a quadratic (C1) union; the sole and heel separate plates under a
  rounded upper edge (the gap line), the sole running back over the heel. Now (heel cup joined at k 35 mm: at 22 a dent showed
  behind the ankle bone): the shaft is the leg plus 1.3 mm of leather
  and over its top 4 mm dives 0.6 mm under the suit, the crossing covered by a piping cord (a seam ridge, no rim, no V); the
  hidden ankle row is taken in at the back 4.5 mm and out at the front 2 mm (the Achilles runs straight into the heel cup;
  the visible ankle moved at most 0.45 mm, 0.232-0.29 m); a superellipsoid heel cup; the foot ONE sweep (rows of half-width,
  top line and exponent along the foot, C2 splines) whose top line rises into the shaft front (the instep one long curve),
  narrowing to an almond toe with a 4 cm toe box, closing as sqrt(1 - t^4); every union a C2 (cubic) smooth minimum; the
  shading normals the field's own gradient; a welt 1.2 mm proud and 5 mm thick and the 2.7 cm block heel (3.2 cm with the
  sole) as their own closed mesh, `Boot_Sole`, with the upper standing 0.8 mm into it (a crisp corner, no gap); a toe-cap
  seam cord. Kept: shaft top 0.232, heel height, foot length (tip -0.171, heel back about 0.058), stance, widths.
  Shaft surface to the leg loft along its section rays (outer / front / inner / back, mm): at 0.13 B1g 4.8 / 5.1 / 4.8 / 4.7,
  now 3.2 / 2.8 / 3.5 / 2.3 (the instep and heel cup begin there); at 0.18 4.3 / 4.4 / 4.5 / 4.4 -> 1.7 / 1.3 / 1.5 / 1.3;
  at 0.215 2.9 / 2.9 / 2.9 / 2.9 -> 1.4 / 1.3 / 1.3 / 1.3 (B1g measured against its own leg). In `side.png` the shaft's depth
  against the leg at 0.25 m: B1g +8 / +11 / +6 px at 0.13 / 0.17 / 0.21 m (back +7.0 / +9.4 / +4.7 mm), now +2 / +4 / +3 px
  (back +2.3 / +2.3 / +1.2 mm). `Boots`: 4 objects (Boot, Boot_Sole, two cords), all closed and outward-facing.
- **The knee pads (B5).** One domed shield (`knee_pad["pad"]`, `bodysuit.build_knee_pads`; dome 9 mm) on the leg's own
  surface, a piped rim, no stitch line (B1g's inset stitch, sampled on the plate's vertex grid, was the dashed oval); the
  drawn frame below it as a seam (from the outer side down to a point on the shin and up to the pad's lower inner edge), the
  outer-shin seam now starting from that point. Front-view width over the knee's width at the pad's widest (geometric): B1g
  0.447, now 0.56 (drawing 0.54 by the review's pixels); height 101 -> 100 mm; top 0.647 (under 0.652). Value in
  `sheet.png` (mean grey, pad / thigh above / shin below): B1g 26 / 29 / 39 and 29 / 37 / 33, now 31 / 29 / 40 and 35 / 37 / 32,
  the drawing 60 / 38 / 35 and 53 / 38 / 38: no longer darker than the suit, still not as bright as drawn (a 12 mm dome got
  the pad to 35-37 but stood 6 mm proud of the knee in the side view, where the drawing's knee is flat).
- **The calf (B8).** `calf` and `calf_top` wider (half-width .056 -> .057, .055 -> .0585) and a little deeper behind
  (.064 -> .065, .067 -> .069); the knee's rows and the loft's tangents at the knee unchanged (nothing moves from 0.54 up), the
  ankle unchanged. Widths in each sheet's frame, render against drawing: back view 0.46 / 0.48 / 0.50 / 0.52 m B1g -3 / -5 /
  -7 / -6 %, now +1 / +1 / -3 / -4 %; front +2 / +5 / +4 / +6 % (B1g -3 / 0 / -1 / +3); side +2 / +4 / +3 / -2 % (B1g +1 / +2 /
  +1 / -2). Measured this way the side was not thin (the review's -7 % side is not reproduced), and the front and back drawings
  disagree on the upper calf: the widening is the compromise.
- Sheet IoU knees down: front 0.908 -> 0.899 (the drawing's high-heeled foot), side 0.856 -> 0.862, back 0.812 -> 0.822.
  `source/mesh_stats.json`: 62 objects, 1,140,986 source / 2,995,708 evaluated triangles, all closed (the flipped-normal list is B1g's).
- Pictures: `boots_before_after.jpg` (B1g above, now below: front, side, back, three-quarter, the vamp, the shaft's top, and the
  reflection lines on the vamp and the side), `knees_before_after.jpg`; every dev view re-rendered (`wrist.jpg` new).
- Reproduce: `--sheet --turn --beauty all --save` (6.9 min; the boots about 80 s of the build),
  `--beauty shoulder,shoulder_back,wrist,boot,flank,flank_back,knee,hand,boot_side,armpit_back,top_close,hand_palm` (5.8 min).
  The before/after pictures and the measures are made by scratch scripts from `Beatrice_b1g_9b6d9df.blend` and this build.

## B1i: the hands and the cuffs, and a leg polish (2026-10-08, answers `previews/review_stage7.md` B2, `review_stage8.md` 1-2)

No reference added. Kit: only Beatrice's own `bodysuit.py` changed; no shared kit file touched (`parts.bare_hand` stays as
it was, unused by her now), so no other character was rebuilt.

- **Step 1, the surface (review B2 "ring-banded").** Cause found by experiment on the B1h hand (scratch harness, the same
  field each time, zebra = reflection lines): (a) as built; (b) the same field Taubin-smoothed, re-projected and shaded
  with the field's own gradient (B1h's boot treatment); (c) (b) with the C2 union everywhere and no knuckle waists. The
  oval and ring patterns in the reflection lines are the same in all three, so the bands are in the FIELD, not in the
  mesh or its normals: `parts.bare_hand` builds the palm as one capsule per pair of rows and every finger as a chain of
  capsules (a node mid-phalanx 6 % thinner), each capsule ending in a rounded cap, unioned with a C1 fillet of 1.2-4 mm,
  plus three pad ellipsoids -- a ring of bump and dent at every node, the boot vamp's B1g cause. Fix at the cause, with
  B1h's method: `bodysuit.FittedHand` -- the palm ONE sweep along the hand (rows as C2 splines, closing round under the
  knuckles, running up the forearm inside the cuff), each finger and the thumb a `_Sweep` along its own bent centre line
  (the bend a sum of C2 steps at the joints; half-width, depths and exponent C2 splines; knuckles and pads smooth
  Gaussian swells of the depth, not nodes; a round tip and a round start, no cut face), every union `smin3` (C2), the
  shading normals the field's gradient, Taubin + re-projection. The hand mesh is built in the hand's frame and its faces
  turned the right way out before the normals are set (the frame is left-handed: `garment.orient_outward` used to flip
  the old hand afterwards, which would have inverted custom normals). Nails: the hand's own mesh, material
  `beatrice_nail` (`PALETTE["NAIL"]`), a plate 0.2 mm proud. The step-1 table (`FITTED_HAND`) is the old hand's layout
  in the new construction. Two traps met on the way, both fixed in the code: a sweep evaluated only inside its bounding
  box must have a box larger than the fillet it is unioned with (a 4 mm margin under a 16 mm fillet drew a jagged crease
  down the thumb side), and a sweep's start must be rounded like its tip (a flat cut face inside the palm showed as a
  line across the back of the hand).
- **Step 2, proportions and form (review B2).** `FITTED_HAND` rewritten: the knuckles 13 mm nearer the wrist (l 0.050-0.058,
  were 0.063-0.071), the fingers longer to match (index 87.5 mm knuckle to tip, middle 95.5, ring 89, little 70.5), so the
  wrist-to-fingertip length in the sheet's front view stays 147-148 mm (B1h 147.9); the palm tapers from 56 mm across
  the knuckles to 32 mm at the wrist, and the wrist is 3-4 mm slimmer in the front view a few mm under the cuff (28.6 mm
  at 4 mm, B1h 32.1); the thumb's ball (thenar) and the heel under the little finger as broad smooth swells, a palm
  hollow, knuckles (a 6 % swell of each digit's back over its joints), finger pads (10 % swell of the palm side mid-
  phalanx), nails. Measured by `C:/tmp/beatrice_b1i/meas.py` on a 4x render of the sheet's front view (numbers in the
  final B1i record below).
- **Step 3, the pose (review B2 "no thumb reads", "mannequin").** Relaxed as in `without_hair*.png`: a curl cascade from the
  index to the little finger (total bend 0.72 / 1.22 / 1.53 / 1.72 rad), the ring and little fingers rolled 0.10 / 0.25 rad
  so they curl towards the thumb's base (no splayed little finger; in the side view the fingers fan forward as drawn),
  the thumb hanging in front of the index, slightly flexed, its nail facing forward. The palm's end follows the knuckles'
  arc (`palm_arc`: 9 mm shorter at the little finger, 5 at the index). Separation for the game-ready step's 6 mm fusing
  (`C:/tmp/beatrice_b1i/gaps.py`, surface to surface): index-middle >= 7.6 mm from 56 mm along the index (5.9 at 47,
  14.7 at the tip), middle-ring >= 7.5 from 61 mm (13.8 at the tip), ring-little >= 7.2 from 47 mm (15.8 at the tip),
  thumb-index >= 7.6 from the thumb's 62 mm (5.7-6.2 at its knuckle, the web, 14.4 at the tip); nearer the knuckles the
  fingers touch, as drawn. Mirror-symmetric (a separate right hand would cost a second hand build, ~2 min; not done).
  Two more surface traps fixed in `_Sweep` / the palm: a digit's start is now a superellipsoid cap (a section shrunk to
  nothing at the start drew a jagged crease through the thumb's fillet), and the distance's scale is the same on the back
  and palm halves of a section (a scale that jumped at the halves' seam left the shape whole but drew a dotted line in
  the gradient normals down the side of the palm).
- **Step 4, the cuff (review B2 "a hard double-ringed tube with a dark gap").** Causes: the band (`band_on`, 0.937-0.950,
  bevelled) began 2 mm above the sleeve's own open end, whose solidified rim showed below it as the second ring; and the
  wrist inside was a flat oval (22 x 32 mm) in a round sleeve end (33 x 34), open on the back and palm sides. Now
  `bodysuit.build_fitted_cuffs` (`garment["cuff_fit"]`): ONE closed ring swept round the sleeve's end, built after the
  hands -- its top where the band's was (0.950, a flat step diving 0.6 mm under the sleeve, so nothing above it moves),
  a 0.7 mm rounded top edge, the outer face 1.7 mm proud of the sleeve (the band's lift + thickness), a 1.3 mm rolled
  bottom edge 1.2 mm below the sleeve's end (covering its rim), and the underside running in to the wrist's skin (found
  on the hand's own field per angle) and 0.5 mm into it: no gap on any side. The hand's rows inside the cuff are round
  and inside the sleeve with a margin (a squarer, wider row poked 0.4 mm through the sleeve just above the cuff, the
  light fleck in B1h's and this round's first front views; now every hand vertex above the cuff lies inside the sleeve).
  `DEV_VIEWS["wrist"]` is now aimed at the left wrist and cuff (B1h's showed the crotch).
- **Step 5, leg polish (review_stage8 corrections 1 and 2; 3-5 not done, as briefed).**
  - Knee pad: the dome 9 -> 3.6 mm over a 1.4 mm edge (was 1.8): the crest about 5 mm proud of the knee; below 0.2 of
    its half-height under the centre its edge eases (C2) down to 0.25 mm at the point (`knee_pad["pad"]["lower"]`, through
    `bodysuit._SinkingAnchor`, a wrapper round the leg anchor that `garment.patch` reads -- `garment.py` untouched), so its
    lower part runs into the shin; the rim cord follows the edge (it sinks with it at the bottom). Its own finish:
    materials `beatrice_pad` / `beatrice_pad_rim` (`PALETTE["PAD"]` #26262c, `["PAD_RIM"]` #4a4a52; gloss in
    `bodysuit.materials`' `pad_*` GLOSS keys, defaults rough 0.22, coat 0.8, coat rough 0.08, specular 0.6).
  - Boot shaft: the ankle bones under the leather (`boot["fitted"]["ankle"]`: smooth Gaussian swells of the boot's field,
    5 mm, the outer at 0.180 m and 3 mm back of the centre, the inner at 0.192 and 6 mm forward) and a 1 mm hollow across
    the back above the heel cup (the leather is 1.3 mm over the suit there, so the suit cannot show through). The top
    cord and the heel cup are unchanged (the swells are under 0.02 mm at the cord). Front width in the sheet's frame
    (sheet px, z 0.16 / 0.17 / 0.18 / 0.19 / 0.20): B1h 49.0 / 48.5 / 48.5 / 48.0 / 48.0, now 49.5 / 50.5 / 53.0 / 53.0 /
    51.0 (the drawing's 56 at 0.19 is in a high heel).
- **The B4 question (upper-arm / deltoid gloss ripples): the same family, a different instance, NOT fixed.** On the B1h
  build, reflection lines on `Suit_Shoulder` (the shoulder patch, meshed from `Suit.union_field`) and `Suit_Sleeve` (a loft):
  the ripples are on the patch only (the sleeve below the arm seam is clean), and with the patch's own field gradient as
  shading normals they get WORSE -- so, as on the hand, they are in the field, not the mesh. The field's arm half is
  `kit.loft_sdf(self.F.arm, ..., m=500)`, the distance to the NEAREST of 500 sampled sections (~0.9 mm apart along the arm),
  which steps at every section; the union with the body is the C1 `implicit.smin`; Taubin smoothing without re-projection
  hides part of it. The fix for the next round is the same as the hand's: a continuous arm distance (the arm loft's own
  centre line projected continuously, as `_Sweep.coords` does), a C2 union, re-projection and the field's gradient as
  normals. Picture: `previews/sleeve_ripples_cause.jpg`.
- **Measured** (`C:/tmp/beatrice_b1i/meas.py` on a 4x render of the sheet's front view, her left hand; the drawing the same way
  at 1x, skin by colour, so the drawing's dark outline is not counted):
  - Wrist (cuff edge) to fingertip: B1h 147.9 mm, B1i 145.0 mm, drawing 136.6. The fingertip's row is B1h's within 0.3 mm;
    the 2.9 mm is the cuff's new rolled lower edge, which reads 3 mm lower in this view (along the arm it is 1.2 mm below the
    sleeve's end, which B1h's band left showing).
  - Wrist width just under the cuff: 0.6 / 4.1 mm below it, B1h 30.7 / 32.1 mm, B1i 27.5 / 30.1 (-3.2 / -2.0). The drawing
    measured this way is 33.6 / 34.7 -- wider than both: the review's 30 vs 27 px was read another way (its rows or the drawn
    outline); by this method B1h's wrist was not thicker than the drawing's. Slimmed as asked.
  - Finger share of the hand (the knuckle line to the tip along the hand, of the cuff edge to the tip): B1h 44 %
    (knuckles 83 mm below the cuff edge), B1i 51 % (71 mm; the drawing 52 % by the review). Measured on the model's own
    table along the hand's axis, which hangs within 6 deg of vertical in this view.
  - Finger separation (surface to surface): see step 3; past the first finger joint every gap is over 7 mm.
- **Nothing else moved** (`C:/tmp/beatrice_b1i/cmp2.py`, every evaluated vertex of every object, `Beatrice_b1h_73ab55b.blend`
  against this build): 56 of the 62 objects are identical (max |dP| < 1e-7 m), the suit's body, sleeves, shoulders, collar,
  zips, piping, panels and the head among them; moved are `Hand_Hand`, `Suit_Cuff`, `Suit_Knee_Pad`, `Suit_Knee_Rim`, `Boot`
  (the ankle bones and the hollow, z 0.004-0.232) and `Boot_Piping_0`, the shaft's top cord, by 0.019 mm (it is laid on the
  boot's surface, and the bones' Gaussian tails reach it at that size). `Boot_Sole` and the toe-cap cord are identical. The
  suit's own surface did not move anywhere (the pad's sinking edge goes into it; the suit does not give way).
- Sheet IoU unchanged to the third decimal except front whole 0.887 -> 0.888, knees down front 0.899 -> 0.900, side whole
  0.902 -> 0.904 (above the waist / waist-knee / knees down: front 0.947 / 0.840 / 0.900, side 0.918 / 0.920 / 0.865, back
  0.914 / 0.837 / 0.822). `source/mesh_stats.json`: 62 objects, 1,173,166 source / 3,058,500 evaluated triangles, all closed;
  the flipped-normal list lost `Hand_Hand` (its faces are built the right way out now).
- Pictures: `hands_before_after.jpg` (the drawings, B1h and B1i in front, side and back views at the drawings' scale x4;
  three-quarter and close-up; reflection lines; matte from her left; the palm side), `legs_polish_before_after.jpg`,
  `sleeve_ripples_cause.jpg`; the whole set re-rendered (sheet, turnaround, beauty, every dev view; `wrist.jpg` re-aimed).
  Not re-rendered: `shape_vs_concept.jpg` (the `--compare` run, 13 min; it still shows B1g's hand), the `--hips` pictures,
  `crimson_hero.jpg`.
- Reproduce: `bash charkit/run.sh Beatrice --sheet --turn --save --beauty hero,hero_back,torso,torso_back,legs,side,head,top,elevated,shoulder,shoulder_back,wrist,boot,flank,flank_back,knee,hand,boot_side,armpit_back,top_close,hand_palm`
  (9.9 min: the build 6.0 min, of it about 1.9 min the hand at 0.7 mm (built once, mirrored), 1 s the cuff, about 1 min the boots and pads).
  The before/after pictures, the measures and the vertex check are scratch scripts in `C:/tmp/beatrice_b1i/` (`hh.py` hands
  harness, `lh.py` legs harness, `meas.py`, `gaps.py`, `cmp2.py`, `compose_*.py`), run on `Beatrice_b1h_73ab55b.blend` and
  this build.

## B1j: the hands, second pass, and the knee pads' finish (2026-10-08, answers `previews/review_stage9.md` hands, items 7-8)

No reference added. Kit: only Beatrice's own `bodysuit.py` changed (by adding: every new behaviour is keyed on a new table
entry, and B1i's table, kept as `FITTED_HAND_B1I`, still builds B1i's hand); no shared kit file touched, so no other character
was rebuilt. `FITTED_HAND` is now B1i's table with the B1j overrides (`beatrice.py`).

- **Step 1, the pose (review 9 "Pose: claw").** The fingers hang close together in a small fan: total curl index / middle / ring /
  little 0.74 / 0.92 / 1.04 / 1.13 rad (B1i 0.72 / 1.22 / 1.53 / 1.72 -- the cascade turned the ring and little fingers out
  sideways, nearly horizontal in the front view), spread over the three joints about 0.38 / 0.44 / 0.18; splay 0.06 / 0.015 /
  -0.03 / -0.075 (B1i 0.17 / 0.07 / -0.04 / -0.09); rolls 0 / 0 / 0.05 / 0.12. The thumb as B1i's, in front. Two attempts: the
  first (0.55-0.92 rad) hung too straight, a stack of parallel fingers in the front view; the second is the one kept.
  **Finger gaps** (`C:/tmp/beatrice_b1j/gaps.py`, each digit's own surface to its neighbour's, by arclength from the knuckle):
  index-middle 0.6-0.7 mm from 9 to 27 mm, 1.9 at 45, 4.2 at 63, 6.3 at 82 (the tip); middle-ring 0.5-0.7 to 29 mm, 2.8 at 49,
  4.5 at 69, 8.4 at the tip; ring-little 0.5-1.3 to 27 mm, 3.6 at 36, 6.6 at 54, 16.5 at the tip; thumb-index >= 6.8 mm everywhere.
  The fingers touch along their first phalanx, as drawn (the union's 1.5 mm fillet leaves a hair-line between them). **The
  game-ready step fuses at about 6 mm, so this pose would come out as a mitten to about two thirds of the fingers' length: the
  game step will need its own finger pose (a `FITTED_HAND` variant with B1i-like splay, built for the bake only) -- not done
  here, since this round builds the drawn pose.**
- **Step 2, the fingers on the hand (review 9 "a ledge", "tubes", "flat and wide").** Cause of the ledge: the palm sweep's back
  stood 1 mm above the knuckles and closed over them as sqrt(1 - t^4), a blunt dome end overhanging the finger roots. Now the
  palm's end is sheared (`palm_end_shear` 0.5: the back side ends about 3 mm sooner, the palm side about 4 mm later, under the
  webbing), closes softer (`palm_end_p` 2.5, over 16 mm), its back lower at the end (depth 6.6 mm, was 7.6), so the knuckle
  heads (`mcp` 0.10) stand out of it; the hand narrower (knuckles 51.7 mm across, B1i 56; palm rows 0.8-2 mm narrower) and
  rounder (exponents 2.2-2.3, were 2.4-2.5), with faint extensor tendons from near the wrist to each knuckle
  (`FittedHand.tendons`: 0.35 mm, a sum of ridges -- a max of them creased where two crossed). The fingers thinner (half-width
  7.2-7.4 mm at the root, 4.0-4.7 before the tip; B1i 6.4-7.8 / 4.4-5.4), closing into a half-elliptic tip (`tip_p` 2; B1i 4,
  blunt), with a 7 % width swell and a 6 % back crease (1 mm wide, 0.5 mm past the joint) at each finger joint (`joint`,
  `crease`). Two attempts: the first (shear, thinner, joints at 5 % / 3.5 %) left the joints invisible and the back flat; the
  second added the tendons and the stronger joints.
- **Step 3, the nails.** Each nail is its own small closed shell lying on the finger (`FittedHand.nail_shells`, object
  `Hand_Nails`, material `beatrice_nail`): an oval (superellipse 2.6) from 11 mm to 2.2 mm before the tip, 0.76 of the
  finger's half-width where it stands (it narrows into an almond with the tip), 0.30 mm proud at the centre and 0.17 at the
  edge (a dome on the finger's own curve), its edge a short wall into the finger. B1i's nails were a material assigned per
  face of the hand's 0.7 mm mesh (a staircase outline) on a 0.2 mm plate; a first B1j try with a domed plate in the field
  kept the staircase, hence the shells. Colour `PALETTE["NAIL"]` #f2c9c1 (B1i #f1c2bd read mauve end-on), gloss in `GLOSS`
  `nail_*` (rough 0.20, coat 0.55 at 0.07; the skin is 0.48-0.60). Checked from the front, her left, behind, three-quarter,
  below and the thumb (`hands_b1j.jpg`, last row).
- **Step 4, wrist and cuff.** The wrist sat 4-5 mm towards the palm inside the round cuff (under the cuff's lower edge the
  skin lay 9 mm inside the sleeve's line on the back of the wrist and at it on the palm side; in the front view the cuff
  overhung the wrist on the outside): the palm rows inside and under the cuff are centred (b +4.2-5.2 mm, w -0.7-1.5 mm), the
  wrist rounder (depth 23.4-24.4 mm a few mm under the cuff, B1i 20) and its back line eased into the hand. The gap under the
  cuff's lower edge is now -0.7..5.3 mm (mean 2.4; printed in the build log). The cuff (`cuff_fit["hug"]`): from 2 mm under its
  top edge the band eases (C1) 35 % of the way in towards the wrist's oval (its first three harmonics: the raw offsets drew a
  crumpled cuff), and its lower edge rolls onto the skin as a quarter ellipse 2.2 mm tall, as wide as the gap left, ending
  0.3 mm proud and running on into the skin: no flat underside. Its top, top edge and step under the sleeve are B1i's; the
  sleeve did not move. Found on the way: the cuff's skin search (`skin_v`) had inside and outside swapped and always returned
  25 mm in, whatever the wrist, so B1i's underside ran flat right through the wrist (the broad rim the review saw from below);
  fixed. Measured on `sheet.png` the review's way (per row, pixels differing from the background; her left / right hand):
  narrowest wrist under the cuff B1i 25 / 25 px, B1j 27 / 28, drawing 30 / 30 (with its outline, about 28 without); cuff /
  wrist B1i 1.44 / 1.48, B1j 1.33 / 1.29, the drawing 1.27 / 1.23 by the same method (the review's 1.15 was read another way);
  perpendicular to the forearm on a 4x ortho (`cuffw.py`) B1i 1.47, B1j 1.11. In the front 4x ortho (`meas.py`, skin only) the
  wrist 0.6 / 4.1 mm under the cuff B1i 27.5 / 30.1 mm, B1j 31.3 / 33.3 (the drawing 33.6 / 34.7): more than the briefed
  1-2 mm in this view, because most of it is the recentring, which the front view sees in full.
- **Step 5, the knee pads' finish (review 9 item 8).** `PALETTE["PAD"]` #141418 (B1i #26262c, the grey base), `["PAD_RIM"]`
  #6c6c76 (was #4a4a52), gloss `GLOSS` `pad_*` rough 0.15, coat 1.0 at 0.07, specular 0.8 (glossier than the boots). The
  shape is B1i's (both pad objects vertex-identical). In `sheet.png`, grey levels on each knee (pad box / thigh box above,
  50 x 80 px): mean drawing 51 / 44 and 47 / 43, B1i 39 / 28 and 48 / 35, B1j 35 / 28 and 45 / 35; the pad's median B1i 37 /
  45, B1j 33 / 36 (the base nearer the suit's); its 95th percentile (the highlight and rim) drawing 115 / 89, B1i 59 / 97,
  B1j 63 / 110.
- **Hand measures** (`sheet.png`, the review's way): top skin row to lowest fingertip B1i 130 / 132 px, B1j 134 / 134, drawing
  128 / 129; the fingertip's row is B1i's (828): the 3-4 px are the cuff's rolled edge, which shows more wrist. Finger share
  on the table (centre lines along the hand: the knuckle line to the lowest tip, of the cuff's edge to it): B1i 54 %, B1j 54 %
  (82 of 152 mm). The hand still hangs about 2 cm lower than drawn, with the cuff (review 7 B7, not this round's).
- **Nothing else moved** (`C:/tmp/beatrice_b1j/cmp2.py`, every evaluated vertex of every object, `Beatrice_b1i_1877dc8.blend`
  against this build): 60 of B1i's 62 objects identical (max |dP| < 1e-7 m), the knee pads and rims among them; moved
  `Hand_Hand` and `Suit_Cuff` (its outer face below the top edge, its lower edge and its hidden inner return); new
  `Hand_Nails`. Sheet IoU front 0.888 -> 0.890, side 0.904, back 0.858 (front hip-knee band 0.840 -> 0.844, back 0.837 ->
  0.838: the hands). `source/mesh_stats.json`: 63 objects, 1,180,138 source / 3,072,444 evaluated triangles, all closed; the
  flipped-normal list is B1i's.
- Pictures: `hands_b1j.jpg` (the drawings, B1i, B1j: front, side and back at the drawings' scale x4; three-quarter and
  close-up; reflection lines; matte and palm side; the cuff close; the nails), `pads_b1j.jpg`; the full set re-rendered
  (sheet, turnaround, beauty, every dev view). Not re-rendered: `shape_vs_concept.jpg`, the `--hips` pictures,
  `crimson_hero.jpg`; `hands_before_after.jpg` stays as B1i's record.
- Reproduce: `bash charkit/run.sh Beatrice --sheet --turn --save --beauty hero,hero_back,torso,torso_back,legs,side,head,top,elevated,shoulder,shoulder_back,wrist,boot,flank,flank_back,knee,hand,boot_side,armpit_back,top_close,hand_palm`
  (10.2 min: the build 6.4 min, of it about 2.1 min the hand at 0.7 mm with its nails, 3 s the cuff). The before/after
  pictures, measures and the vertex check are scratch scripts in `C:/tmp/beatrice_b1j/` (`hh.py` hands harness, `lh.py` legs
  harness with `--mat`, `meas.py`, `sheetmeas.py`, `cuffw.py`, `gaps.py`, `share.py`, `cmp2.py`, `compose_b1j.py`), run on
  `Beatrice_b1i_1877dc8.blend` and this build.

## B1k: the upper body's surface -- shoulders, upper arms, collar, armpits, inserts (2026-10-08, answers `previews/review_stage9.md` items 1-6)

No reference added. Kit: only Beatrice's own `bodysuit.py` changed, by adding (every new behaviour keyed on a new table entry:
`shoulder["field"] = "c2"`, `join_curve`, `mesh_box`, `box_c2`, `arm_form`, `kfac`; `collar["flare"]`; `seams["shoulder"]`;
a panel's `shared` and `res`; a seam point `("panel", name)`; GLOSS `mesh_width`); no shared kit file touched (`wearmat.mat_mesh`
already took `width`), so no other character was rebuilt. B1j's tables are kept where they changed (`ARM_SEAM`,
`RAGLAN_FRONT_B1J`, `RAGLAN_BACK_B1J`; the old collar `rings`).

- **Step 1, the shoulder and upper arm (review 9 items 1-2).** The cause found in B1i, fixed at the cause with the hand's and
  boot's recipe. `Suit.arm_proj`: the arm's distance continuous -- each point's own section plane found by Newton on
  (P - c(t)) . T(t) = 0 from the nearest of 160 centre-line samples, the radial gap scaled by the section's slope (B1j:
  `kit.loft_sdf`, the nearest of 500 sections, stepping every 0.9 mm). The union C2 (`smin3`, the fillet 1.5x wider so its bulge
  is B1j's). The patch meshed in a band round the surface (`_mesh_banded`), trimmed, its edges laid exactly on the seam lines,
  Taubin-smoothed with the edges held, every vertex re-projected onto the field (`Suit.patch_field`: the union with the bust's
  form folded in where the torso rules -- B1j displaced afterwards and lost it on the cut's re-projection), the field's gradient
  as the shading normals. Two attempts: the first fixed the field (the mottle and the streaks gone, reflection lines in
  `shoulders_b1k.jpg`); the rest of the round went into the joins it exposed (below).
  - **The seams as drawn.** The three drawings agree on a V round the upper arm at the foot of the deltoid, lowest on the outside
    and rising to the armpit front and back (the review read it the other way round); the front and side views differ on its
    height at the arm's front (1.316 against 1.346): the front view's ruled, as briefed. `DELTOID_SEAM`: on the surface 1.287
    outside, 1.316 at the arm's front, ending ON the raglan seams' cords in front (1.334) and behind (1.330); drawn on the
    shoulder's own surface (`build_shoulder_cords`). The elbow seam (`ELBOW_SEAM`, about 1.16, a soft V behind, as drawn) is
    the patch's lower edge: the shoulder's own surface now runs down to the elbow (`mesh_box`), and B1j's horizontal ring at
    mid upper arm -- its old lower edge -- is gone. A soft fold over the elbow seam, mostly behind (`arm_form["fold"]`).
  - **Fuller.** `arm_form`: the upper arm 5 % of its radius fuller from the elbow seam up, gone by the shoulder's top; a rounder
    deltoid (1 mm, outside). The sleeve below the elbow seam is B1j's loft (the arm loft's rows unchanged). Upper-arm width in
    the sheets' frames (px, `armw.py`, outermost run of figure pixels per row): front view z 1.198-1.244 B1j 50-52, B1k 53-54,
    drawing 54-58 (+4..+8 %); back view z 1.198-1.246 B1j 49-51, B1k 52-53, drawing 52-59. The shoulder: front-view half-width
    at 1.36 178.8 -> 179.4 mm, at 1.34 unchanged (182.9), the widest across the deltoids (1.30) 188.7 -> 189.8 (drawing 186.9;
    a first pass, swell held to 1.29 and a 1.6 mm deltoid, made it 192.1 -- toned down). Forearm, cuff and hand unmoved.
    The reflection lines on the deltoid and the back of the upper arm are smooth (`shoulders_b1k.jpg` rows 2-3): the
    orange-peel and the streaks are gone.
  - **The armpits** (with step 3): see below.
- **Step 2, the collar (review 9 item 3, review 7 B3).** `CollarFlare`: in each half-plane through the neck the collar's profile
  is the C2 smooth union of a tube that hugs the neck (1.5-2.5 mm off the placeholder neck) and the body's own section, the
  body's neck taken away from 6 mm (side) / 1.5 mm (front) / 3 mm (back) outside the tube on (its own curve continued
  straight, C2), so the union flares into the trapezius with no edge and does not bulge where the body's neck runs up inside
  the collar; the fillet 34 / 16 / 24 mm wide at the side / front / back. Its top rolls in onto the neck over 1.8 mm, the edge
  inside the neck: no lip. Where it rises out of the body it lands tangentially and shades with the body's own normal,
  blended into its own over 0.8 mm of height (a jump in the shading normal is what showed as the ledge). Height to 1.482 as
  before; the zip runs up it (the body's zip is B1j's, cut off where the collar lands; the collar's own on the flare). The
  raglan seams and the back's centre seam start a centimetre up under the flare, so their cords come out from under the
  collar. Attempts: 9 short harness runs (C1 bends at the body's crossing drew a crease; per-angle steps drew pleats in front;
  a crossing at 0.3 mm under the body drew a line). Front-view half-width (mm, outline edges / 2): at the top 1.468-1.482
  drawing 39.9, B1j 42.2-44.0, B1k 42.8-44.0 (the placeholder head's neck is 40-42.5 there: the collar cannot be narrower than
  the neck it hugs); the flare at 1.454 / 1.447 / 1.440: drawing 48.6 / 59.6 / 75.2, B1j 47.5 / 55.0 / 76.4, B1k 49.8 / 57.9 /
  76.4. Side view half-depth at 1.454-1.468 B1j 49.3-51.7, B1k 47.6-48.7, drawing 39.3-42.9 (the neck again: 43).
- **Step 3, armpits, piping, inserts (review 9 items 4-6, review 7 B6).**
  - *Doubled lines.* Each raglan seam now comes from under the collar higher over the shoulder, 1-2 cm clear of the insert's
    upper half (as drawn), converges on the insert's outer edge and RUNS ALONG IT to its lower tip (`("panel", name)` splices the
    insert's own smoothed outline into the seam; the insert's frame leaves that stretch out, `shared`): one piped line, no
    zigzag. The front and the back alike.
  - *The back-armpit tear.* Two causes. (1) B1j's region test (x past the cut's x) sliced through the armpit's folds; the
    shoulder's own surface is now a sector of the body's level sections between the two raglan lines (`Suit.in_torso_region`,
    `cut_target`), every body vertex inside it moved onto the nearest line along the loft, the patch's ragged edge laid onto
    the same lines the same way. (2) Behind the arm the arm's back overlaps the torso 4-8 mm beyond the raglan line: the body
    loft, lifted that far along its normal into the fold, tore (8 mm cap: a trench) or folded over the line (16 mm). Now the
    shoulder's own surface reaches past either line as far as the union lifts the torso at all (`Suit._region`: the arm's
    distance under the fillet's reach, a smooth curve in height): beyond it the union IS the torso, so body and patch meet
    there with no step and the body is never lifted into a fold. A non-manifold saddle of the surface nets where the gap
    between arm and torso closes left a small hole: filled (`_fill_small_holes`). The box beside the neck from x 0.03 (the
    patch's ragged edge on the box's face at 0.05 was the teeth over the blade insert's upper tip). Cords and inserts on the
    shoulder's surface are laid onto it by Newton steps to its nearest point (`Suit.lay`); the shoulder inserts triangulated
    at 1.5 mm (at 4 mm they sank under the shoulder's own mesh in the fold).
  - *The mesh.* `GLOSS`: cell 3.6 -> 2.2 mm, thread share (`mesh_width`) 0.30 -> 0.06 of the lattice's ramp (B1j's net was 83 %
    thread: a dark sheet perforated with light dots), the skin under it 0.42 -> 0.48, the clear coat 0.35 -> 0.10: a fine dark
    net with warm skin through it. The shapes and places of all inserts kept. Mean sRGB in 7x7 px boxes on the chest inserts in
    the front sheet: drawing (123, 101, 101) / (120, 103, 102), B1j (81, 75, 74) / (89, 84, 83) grey, B1k (118, 104, 99) /
    (128, 116, 111).
- **Nothing else moved** (`C:/tmp/beatrice_b1k/cmp3.py`, every evaluated vertex of every object, `Beatrice_b1j_8290f51.blend`
  against this build; an object whose vertex count changed is checked by each vertex's distance to the other mesh's surface):
  22 objects identical (max |dP| < 1e-7 m): the hands and nails, the cuffs, the boots and their cords, the knee pads and rims,
  the head, the waist, hip, back and thigh inserts, the zip stops. `Suit_Body` changed only between z 1.281 and 1.448 (the
  armpits, the shoulders and the neck under the collar; below 1.281 vertex for vertex B1j's). `Suit_Sleeve` ends at the elbow
  seam now: every one of its vertices lies on B1j's sleeve within 0.015 mm (the faceting of B1j's mesh: the same loft surface).
  The body zips are B1j's vertex for vertex, cut off where the collar lands (the cut row's solidify rim is new); the collar's
  zips and the pulls moved with the collar. Moved as briefed: the shoulder's own surface, the collar, the two shoulder inserts
  and the piping (THE line's path now starts higher, so its cord is resampled down to the hip; the lines below the armpit are
  the same curves). Sheet IoU above the waist: front 0.947 -> 0.951, side 0.918 -> 0.921, back 0.914 -> 0.917; the other bands
  unchanged. `source/mesh_stats.json`: 64 objects (new `Suit_Piping_Shoulder_0`), 1,241,980 source / 3,265,260 evaluated
  triangles, all closed; the flipped-normal list gained the two zip pulls' tabs (boxes anchored on the new collar).
- Pictures: `shoulders_b1k.jpg`, `collar_b1k.jpg`, `inserts_b1k.jpg`; the full set re-rendered.
- Reproduce: `bash charkit/run.sh Beatrice --sheet --turn --save --beauty hero,hero_back,torso,torso_back,legs,side,head,top,elevated,shoulder,shoulder_back,wrist,boot,flank,flank_back,knee,hand,boot_side,armpit_back,top_close,hand_palm`
  (9.3 min: the build 5.5 min, of it about 100 s the shoulder and upper arm -- meshed in a band, 43 s; re-projection and
  normals 50 s -- and 6 s the collar). The before/after sheets, measures and the vertex check are scratch scripts in
  `C:/tmp/beatrice_b1k/` (`sh.py` suit harness, `ch.py` collar harness, `zb.py` gloss / reflection-line / object-colour views,
  `armw.py`, `meas_b1k.py`, `cmp3.py`, `compose.py`), run on `Beatrice_b1j_8290f51.blend` and this build. Not re-rendered:
  `shape_vs_concept.jpg`, the `--hips` pictures, `crimson_hero.jpg`; `sleeve_ripples_cause.jpg` stays as B1i's record.
