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

## B1l: the finishing round before the full body pass (2026-10-08, answers `previews/review_stage10.md` items 1, 2, 5, 7, 8, 10; item 3 prepared for the owner)

No reference added; the body's shape unchanged. Kit: only Beatrice's own `bodysuit.py` and `suitbuild.py` changed, by adding (every new
behaviour keyed on a new table entry: `shoulder["root_fade"]`, `["cut_nearest"]`, `["sharp_open_edges"]`, `["region_smooth"]`; a seam spec's
`back_line`; a knee pad's `profile`; `tables.gloss(variant)` with a variant's `gloss`); no shared kit file touched, so no other character
was rebuilt. B1k's tables kept where they changed (`RAGLAN_BACK_B1K`, `DELTOID_SEAM_B1K`, the old seams and fold in comments).

- **Step 1, the back-armpit knot (item 1; items 5 and 10's doubled lines).** Three causes, each fixed where it arose.
  1. *A cliff in the field.* `Suit.arm_proj` gave every point past the sleeve loft's ROOT end (its plane, through (0.104, 0.027, 1.35), comes
     out of the torso behind the armpit from the fold at 1.30 up beside the blade insert to 1.42) the distance 1: on one side of that line
     the union lifted the torso 2-4 mm, on the other not at all. The body loft folded over it (dihedrals of 30-55 deg along the shoulder's
     edge) and the fold's V had a crease through it. Now the arm runs on past its root as the root's own section plus a C2 ramp,
     w (s / w)^3, w = 25 mm (`root_fade`): the lift fades over about 2 cm, and the shoulder's own surface reaches as far as it does
     (its edge behind the armpit moved from about -34 to -46 deg of section angle at 1.33).
  2. *A cut through the saddle.* The shoulder's own surface ended on the torso at a level line at RAGLAN_LOW (1.293) -- inside the
     union's reach (it lifts the torso down to 1.278; the armpit's saddle is at 1.300), so the fillet's back face ended in a horizontal
     edge: the flap. It now reaches down to `CUT_LOW` 1.268: every edge of the shoulder's own surface on the torso lies where the union IS
     the torso, so no join there needs a cord, and the short cord under the arm is gone.
  3. *The rims in the shading.* Every vertex on an open edge was shaded with the average of its outer face and its solidify rim (45 deg
     off the surface, measured on the evaluated mesh): a staircase of dark teeth along every join with no cord over it (the patch's
     edge behind the armpit; B1k's ragged edge and dark notches). `_sharp_open_edges` marks the open edges sharp (off the mid-plane)
     on the body, the shoulder's own surface, the sleeve and the collar; positions untouched; the teeth gone.
  - *The cords as drawn.* `BACK_LINE`: ONE line down the blade insert's outer edge, round into the armpit and on down the back past the
    rib and waist inserts (B1k: the raglan seam ran on straight to 1.293 past the insert's tip while the back seam started beside it at
    1.326 -- two cords crossing in an X -- and the under-arm cord ran across their feet). The deltoid seam ends ON that line at 1.312
    (`back_line`; B1k at the insert's tip at 1.330 through a last point in the fold: the hook). The raglan seam meets the blade insert's
    edge at its middle point at a clear angle (`shared` (2, 4); B1k ran 4-5 mm outside the edge for 3 cm). The back V starts at the
    insert's upper tip (B1k: from under the collar beside the raglan seam's start, two lines side by side and a hook).
  - The region's edge smoothed wider in height (`region_smooth` (14, 12), B1k (8, 5)): a sharper turn left a hair gap between the body's
    3.3 mm rows and the patch's 1.2 mm edge (a dark crack); the patch's edge vertices beyond the lines go to their nearest point on
    them (`cut_nearest`).
  - Tried and dropped: relaxing the quads surface nets fold in the fold's deep V (70 -> 38 folded quads, re-folded by the re-projection,
    no visible gain, 30 s). The crinkles left there come from the arm's section planes fanning on the inside of the arm's bend at the
    armpit (points 0.5 mm apart find sections 1.4 cm apart along the arm: the arm's distance jumps up to 4 mm) -- a sleeve-field
    question for a later round; they show only in a close-up of the fold (`armpit_b1l.jpg` "fold, close").
  - Proof: `armpit_b1l.jpg` (the drawing; hero_back at 2x, close, three-quarter back, straight back, the fold close; reflection lines;
    the back yoke; the elbow), `armpit_back.jpg`.
- **Step 2, the knee pads (item 2).** `_DomedAnchor`: the pad a crowned dome, 4.4 mm over the 1.4 mm edge as (1 - r^2)^0.6 (its edge
  rolls down steeply under the rim, so the highlight runs round it as a streak), plus a 1.5 mm soft ridge down its middle (`profile`).
  Side-view proudness (the pad's front line minus the suit's, per height, `C:/tmp/beatrice_b1l/kh.py`): B1k max 5.6 mm, B1l max 7.8 mm
  at 0.58-0.60 (target 6-8, on the drawn knee line). The shape alone did not do it: at the B1j finish (coat 1.0, specular 0.8) the face
  threw back the backdrop's grey as one sheet whatever its curvature (tested at 8.3 mm); now coat 0.6, specular 0.5. The knee frame's
  cord ends ON the pad's rim (x .104 at .5685; B1k 1.4 mm inside it, where the pad's lower part sinks: its end cap came up through the
  pad). Proof: `pads_b1l.jpg`, `knee.jpg`, `legs.jpg`.
- **Step 3, small polish.** *Elbow (item 7):* the fold over the seam removed (B1k's 0.6 mm ridge 8 mm above the join, mostly behind:
  the outline bulged just above the seam and stepped in at it); with the sleeve's top edge no longer shaded with its rim, the seam is
  a plain line (`armpit_b1l.jpg` "elbow"). *Thumb (item 8):* the fingers' joint form and taper (half-widths 9.4 / 8.1 / 6.9 / 5.7 mm,
  B1j 9.6 / 8.4 / 7.4 / 6.2; joint swell 7 %, a back crease, a knuckle), curled more (bends 0.08 / 0.28 / 0.32, B1j 0 / 0.15 / 0.22)
  and heading nearer the index (dir (0.94, 0.21, -0.24), B1j (0.91, 0.32, -0.27)): from the front the V between thumb and fingers is
  narrower. Thumb-index surface gap (`gaps.py`; the game-ready step fuses at about 6 mm): B1j/B1k min 6.8 mm, B1l 7.2 mm (tried closer:
  4.2 and 6.0 mm, too near). *Collar base (item 10):* the stair-stepped patches left of the zip are the placeholder head's skin
  reflected in the body loft's 3.3 mm triangles; one attempt (the collar landing 0.2 mm deeper, a wider normal blend) changed nothing
  visible and was dropped: NOT fixed. *Cuff dent (item 10):* the light patch is the face reflecting the backdrop where the band eases in
  over the wrist's flat front; one attempt (the face's own normals) changed nothing visible and was dropped: NOT fixed.
- **Step 4, a look for the owner (item 3).** `VARIANTS["wet"]` (`gloss`: rough 0.30 -> 0.20, coat 0.45 -> 0.85, coat_rough 0.20 -> 0.07,
  specular 0.45 -> 0.55; the boots are 0.22 / 0.70 / 0.10), `tables.gloss(variant)` read by `suitbuild.build`. The default is unchanged.
  `gloss_compare.jpg`: default and wet on the same build, same light (hero, hero_back, legs). Suit pixels in fixed boxes (luminance p50 /
  p95 / p99): hero belly 60 / 92 / 134 vs 61 / 105 / 135; hero_back 68 / 112 / 141 vs 75 / 111 / 139; legs 74 / 112 / 139 vs 86 / 119 / 142.
  The wet suit's reflections are crisper but not brighter at the top: the studio rig has broad soft lights and a large grey backdrop
  and no small hot sources, so a glossier suit mirrors more grey instead of throwing the drawing's small hot streaks. The rig is a large
  part of the satin look; its default is unchanged too.
- **Nothing else moved** (`C:/tmp/beatrice_b1l/cmp3.py` and `cmp4.py`, `Beatrice_b1k_77ed9b4.blend` against this build): 33 objects
  identical (max |dP| < 1e-7 m): the head, the boots and their cords, the collar, the sleeve, the knee rims, every mesh insert but the
  blade's, the zips. Moved as briefed: `Suit_Shoulder` (old vertices off the new surface by > 0.1 mm: 1847 of 129824 -- behind the elbow
  at z 1.15-1.17 up to 0.69 mm, the fold removed, and in the armpit up to 3.5 mm); `Suit_Body` only between z 1.268 and 1.418 behind and
  under the armpit, where the shoulder's own surface now reaches (every new vertex on the old surface within 0.015 mm);
  `Suit_Mesh_Blade` 494 vertices at z 1.359-1.391 by up to 1.6 mm (laid on the shoulder's surface, whose edge moved with the raglan
  line); the piping (the back's cords, the deltoid seam, the knee frame's end); `Suit_Knee_Pad`; `Hand_Hand` the thumb and its ball
  (x 0.26-0.30, z 0.83-0.95); `Suit_Cuff` 0.1-0.35 mm on the thumb side (it is seated on the wrist's skin, which moved at the thumb's
  root). `source/mesh_stats.json`: 62 objects (the under-arm cord and the back seam's second cord gone), 1,247,563 source / 3,292,964
  evaluated triangles, all closed. Sheet IoU unchanged above the waist (front 0.951, side 0.921, back 0.917).
- Reproduce: `bash charkit/run.sh Beatrice --sheet --turn --save --beauty hero,hero_back,torso,torso_back,legs,side,head,top,elevated,shoulder,shoulder_back,wrist,boot,flank,flank_back,knee,hand,boot_side,armpit_back,top_close,hand_palm`
  (9.3 min: the build 5.4 min); the wet look: `--variant wet --beauty hero,hero_back,legs`. Scratch harnesses in `C:/tmp/beatrice_b1l/`
  (`sh.py` suit, `kh.py` knee pads, `hh.py` hands, `ch.py` collar, `zb.py` gloss / reflection lines / object colours, `wet.py`,
  `cmp3.py`, `cmp4.py`, `gaps.py`, `compose_b1l.py`; probes `cliff.py`, `fademap.py`, `nrm2.py`, `tjump.py`). Not re-rendered:
  `shape_vs_concept.jpg`, the `--hips` pictures, `crimson_hero.jpg`.

## B1m: the body accepted -- wet gloss and the concept pose (2026-10-19, the owner's words of 2026-10-19)

The owner: "going very glossy is fine"; the hips "look pretty cool for me now, but it's really hard to tell from these poses";
before finalising, a model in the concept's pose. The body's GEOMETRY does not change in this round.

- **Step 1, the wet gloss is the default.** `GLOSS` takes B1l's `"wet"` values (rough 0.20, coat 0.85, coat_rough 0.07,
  specular 0.55); B1b-B1l's default is `VARIANTS["satin"]` (0.30 / 0.45 / 0.20 / 0.45), rebuilt with `--variant satin`;
  `"wet"` stays as a name (it now rebuilds the default) so B1l's commands still run. The boots, pads and nails keep their own
  values. The full picture set regenerated (`--sheet --turn --save --beauty` with every beauty and dev view, B1l's command:
  9.8 min, the build 5.8 min). `source/mesh_stats.json` unchanged (byte for byte); `sheet_scores.json` moves in the fifth decimal
  (the glossier edges' alpha). The studio rig is unchanged.
- **Step 2, the concept pose.** A render-time step on a COPY of the saved build: `scripts/posed.py` opens
  `source/Beatrice.blend`, poses it and renders; the build file is never written (its mtime unchanged). Mechanism
  `charkit/suitpose.py` (new, Beatrice's own like `suitbuild.py`; no shared kit file touched, so no other character was
  rebuilt); the tables `beatrice.POSE` (joints, weight bands, rigid parts, the pose, the shots, the optional light setup);
  the sheets `scripts/posed_sheets.py` (plain Python + PIL).
  - *Route: a skeleton of our own*, not the game rig. `rig.py` binds the fused ~10k-triangle game skin by bone heat after
    a voxel remesh (`gameready.py`); the high-resolution set is 62 closed parts with 3.3 M evaluated triangles, half of them
    mirrored, and the brief only needs stills, so the cleaner route was 17 bones with world-aligned rest frames (pelvis,
    spine, chest, neck, head; per side upper arm, forearm, hand, thigh, shin, foot; the fingers keep the hand's modelled
    curl) and our own forward kinematics and linear-blend skinning in numpy. Every part's modifier stack is applied first
    (`new_from_object`, every layer kept: the custom normals follow the deformation, the piping and net attributes too).
  - *Weights: ONE smooth field of position* (C2 ramps), evaluated at every vertex of every part, so a cord, the zip, an
    insert or a cuff moves exactly as the suit under it and nothing can float off or sink in: the trunk shared by height
    (pelvis/spine 1.00-1.10, spine/chest 1.16-1.28, chest/neck 1.41-1.46, neck/head 1.48-1.53); an arm owns what lies within
    5-8.5 cm of its own polyline (shoulder-elbow-wrist-knuckle-fingertip) past a ramp across the shoulder joint, the fade
    widened to 16 cm where the arm still meets the trunk; a leg owns what lies below a ±5 cm ramp across the hip joint's plane,
    faded to nothing at the mid-plane over 5 cm (the crotch moves with the pelvis); ramps across the elbow, wrist, knee and
    ankle planes. Rigid (`rigid`): the knee pads and their rims, the zip pulls and stops, each side whole at the field's value
    at its centre. The boots and soles follow the field (the sole lies wholly under the ankle ramp, so it turns with the foot).
  - *The pose* (`POSE["poses"]["concept"]`, world rotations): weight on her RIGHT leg, the pelvis 6 cm to her right, tilted
    9 deg (right hip up) and turned 5 deg; the spine bending back 8 deg, the shoulders tilted 5 deg the other way (the right one
    lower); the left leg free (thigh 22 deg forward, knee about 40 deg, heel lifted); the right arm a little forward and out,
    the left hanging close beside the hip (3 cm from the hip at the nearest); the head nearly level. The figure is lowered so
    the right sole is on the floor (10 mm), the free sole ends 0.1 mm above it. Nearest gaps on the posed mesh
    (`C:/tmp/beatrice_b1m/gaps.py`): left hand to hip 30 mm, right hand 53 mm, legs 52 mm, boots 84 mm.
  - *Camera* (`shots.concept`): 10 deg to her right, at 1.25 m (chest height), 2.6 m away, 90 mm, 1024 x 1536 like the
    concept, framed by the body's own landmarks (the shoulder line and the crotch at the concept's heights). The concept's
    head is drawn far larger (hair to the frame's top): the render's crown sits about 130 px lower.
  - *Two attempts*, judged beside the concept at the same size with the render's outline drawn over it. 1: the right arm
    hung too close (its hand 70 px out from the hip against the concept's 140), the pelvis's sideways shift weaker, the
    shoulders level. 2 (kept): the right arm further out and forward, the pelvis 1 cm further over, the shoulders tilted,
    the right thigh a little further out, the left leg stepping. A fold under the right armpit in attempt 2's first render
    (the flank dragged by the arm through a 3 cm fade) was fixed with the wider fade near the trunk.
  - *Lights*: the default studio rig for `posed_vs_concept.jpg` and `posed_full.jpg`. `posed_hotlight.jpg` adds the optional
    setup `POSE["lights"]["hot"]` (`--lights hot`): the studio's broad lights at 0.45, the world at 0.8, and three narrow
    rectangles (a 6 cm x 0.9 m key strip front right, a 5 cm x 1 m strip at her left, a 12 cm square above) that the gloss
    throws back as thin hot streaks on a darker suit, nearer the concept's latex. The default rig is unchanged.
  - Proof: `posed_vs_concept.jpg` (concept | render | the render's outline over the concept), `posed_full.jpg` (front, front
    three-quarter, side from her right, back three-quarter), `posed_hotlight.jpg` (concept | studio rig | hot). Close-ups for
    checking the deformation are dev shots (`--views hips,seat,armpit,armpit_r,knee`), not committed.
  - Measured on `posed_vs_concept.jpg`'s render (alpha) and the concept (edges read by eye, about ±10 px), full-size pixels,
    same vertical scale: shoulders with the arms render 543 / concept 630; waist 308 / about 352; hips (widest, 1300-1400)
    458 / about 505; hips / shoulders 0.84 / 0.80; hips / waist 1.49 / 1.43; the crotch's offset from the chest's centre
    59 / about 115 px (the concept's sway is about twice the render's).
- Reproduce (from `art/characters`, about 1.3 min):

      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/posed.py
      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/posed.py -- --views concept --lights hot
      python Beatrice/scripts/posed_sheets.py

  (58 s for the five shots, 15 s the hot one; freezing and posing 4 s.) `--draft` for a quarter-size loop (about 6 s).
  The raw RGBA renders go to `$TEMP/Beatrice_posed`.

## B1n: the bust, and the profile (2026-10-19, the owner's words of 2026-10-19)

The owner on `posed_vs_concept.jpg`: "the breasts could use some more work"; the original concept's look natural, the A-pose
drawings' (`without_hair*.png`) "too much" (not a bust reference). References added (supplied by Erik, 1024 x 1536 each, side
views of the upper body in the concept's style; their high-cut suit is ignored, the full-length suit stays):
`concept-upper-body-side-v2.png` (THE profile reference, "the better version"; SHA-256
`6188478674b00e08251d66f59cd583d3a71fdbdc318d0f2484e54cf0c3a58783`) and `concept-upper-body-side-v1.png` (bust "perhaps a
little large", "hips are pretty good"; SHA-256 `cf77784a2806c6b11bfbf531ba7f7adce21ff56e05127c9e372cdc262904d6a9`).

### Step 1, the target (measured before building)

- **Method.** v2's front outline extracted per pixel row (the last pixel differing from its flat grey backdrop by more than 60
  in summed RGB). The model's side silhouette from the B1m build (`Beatrice_b1m_922fc39.blend`, unposed): the front edge of
  every level section of Suit_Body / Shoulder / Collar / Sleeve restricted to |x| < 0.125 (the arms out), every 1 mm. The two
  put in one frame by the similarity that takes two landmarks onto the model's: the collar's base (v2 row 486, the front
  outline's least x between the collar and the chest) onto the neck-base row (1.437 m, y -0.0367), and the front waist (row
  1007, the front outline's least x between the bust and the belly) onto the model's (1.122, y -0.1040): 1650 px/m, a 0.9 deg
  turn. The waist landmark is the front outline's own narrowest point (the front view's waist is 1.138). The vertical
  registration is good to about +-5 mm (where the collar's base is on each); on the upper slope (about 1 mm forward per mm of
  height) that is +-5 mm of depth, on the apex and the fold it is not.
- **v2 (the target)**, mm in front of the body's centre line: the apex 131.3 at 1.277 m, the plateau within 1.5 mm of it from
  1.271 to 1.297 (its middle 1.284); the lower curve full (125.0 at 1.256, 6 mm behind the apex 28 mm under it) and then
  tucking steeply into the fold at 1.240 (114.9; 1.240-1.250 m: 6.7 mm in 10 mm); under the fold the ribs stand straight at
  113-114 from 1.20 to 1.24. The upper slope 129.7 / 117.9 / 100.4 / 78.5 / 58.8 / 47.5 at 1.30 / 1.32 / 1.34 / 1.36 / 1.38 /
  1.40 -- about 45 deg from 1.31 to 1.37, concave into the collar's base above that. v1 by the same method (its own landmarks):
  apex 142.3 at 1.287.
- **B1m (the model)**: apex 151.2 at 1.285 -- 20 mm in front of v2 (and 9 mm in front of v1, which the owner found "a little
  large"); 128.7 at 1.24 (v2's fold: 14 mm in front), 147.9 / 130.6 / 106.5 / 86.1 / 68.2 / 52.9 on the upper slope (8-18 mm in
  front of v2 from 1.30 to 1.38): a cone, its upper and lower slopes nearly straight, no fold. Level sections: at 1.28 the
  centre (the zip) 125 mm, the breasts' front flat at 143-150 from x 30 to 80 mm -- the shelf -- and the valley a 30 mm wide
  V at the zip (the old centre clip, x 6-26 mm).
- **The concept, front three-quarter, through the posed overlay** (`posed_vs_concept.jpg`'s framing: the shoulder line and
  the crotch at the concept's heights): each breast its own round volume reaching from the zip out to the torso's outline,
  its apex about half way (her right: outer edge x 337 px, zip 579; her left: zip 590, outer edge about 790), the lower curves
  lowest at about 925 px (her right) and 905 (her left), each with its own shadow; the zip in a shallow valley between them.
  The render: the outer edges at about 345 / 735 (with the inserts), the underbust seam lowest at 850 / 845 (825 at the zip),
  no separate volumes. So in the concept's own framing its breasts sit 55-75 px LOWER than the model's underbust seam, not
  higher -- but v2 (its own landmarks) puts the fold exactly at the model's seam (1.232-1.254); the concept draws the torso
  longer under the shoulders (its shoulders with the arms 630 px against the render's 543). The height of the fold stays the
  seam's (v2); the volumes go a little wider (apex x 60 -> 72 mm) and become two.
- **Target, as numbers**: the side silhouette within a few mm of v2's from the fold to the upper slope (1.22-1.37 m): apex about
  131 mm at 1.27-1.30, the fold at 1.24 with the lower curve round above it, the upper slope 118 / 100 / 79 at 1.32 / 1.34 /
  1.36. Above 1.39 (the collar's base, where v2's registration is weakest and the collar is laid on the plain loft) nothing
  changes. In front: two volumes with a soft valley at the zip (10-15 mm deep at the apex's height), the apex about 72 mm out.

### Step 2, the bust rebuilt (three attempts)

- **Mechanism** (`charkit/bodysuit.py`, Beatrice's own; no shared kit file touched, so no other character was rebuilt): a
  `dims["forms"]` row given as a dict is a DESIGNED field, `bust_form` (B1b-B1m's tuple rows -- the seat, the old bust -- build as
  before). Two breasts, each `bust_dome`: T (1 - r^2)(1 + c r^2) below the apex (a polynomial in the elliptic radius r -- smooth
  everywhere, negative outside its footprint, fuller than a paraboloid: the round lower curve), blended (C2) above it into the bell
  (1 - r^2)^3 (the long upper slope easing out of the chest), the half-axes changing C2 across the apex (inwards / outwards, below /
  above); the two united across the centre by a C2 smooth maximum (`kv`: the valley at the zip is their fillet); the chest wall's
  C2 smooth maximum with them (`k`: 5 mm under the breasts -- the fold -- 8 mm over them); plus `bust_wall`, the wall taken back
  between and above them (a natural cubic in height inside a C2 window from 1.185 to 1.405 m, times (1 - (x/X)^2)^3 across, X
  0.11); all of it on the front only (a C2 ramp from y -0.05 to -0.01). Every piece is C2 and compact: no line in the reflections,
  nothing outside its reach moves. `Suit.forms` and `Suit.patch_field` (the shoulder's own surface, where the torso rules) read it;
  `_FormsLoft` lays the front zip (tape and teeth) on the displaced surface (it lay on the plain loft, where B1m's forms were cut
  to zero over the centre front; where the field is zero it is the loft vertex for vertex); `Suit.radial_lift` lifts a body point
  near the shoulder box onto the union only past the wall's own depth (it pulled the upper chest's points inside the box back out:
  a jagged fold along the box's face at x 0.03 in attempt 1's first build). Tables: `beatrice.BUST`; B1m's Gaussian is
  `BUST_B1M`, the shape `b1m` (`--shape b1m`).
- **The design loop**: the plain body loft dumped once over the chest (points, normals), the field evaluated in plain numpy on it
  (side silhouette against v2's, level sections, a crude shaded and reflection-line view) -- a second per candidate; then a
  body-only rebuild on the B1m build (35 s) and full builds (5.5 min) judged in close-ups, reflection lines and the posed concept
  view. Shapes tried on the way (numpy only): an ellipsoid cap (its edge's infinite slope drew a line round each breast in the
  reflection lines), the bell all round (hanging teardrops with a crease under them), a short paraboloid (two flat buns with a
  ledge on top).
- **Attempt 1** (apex (72, 1287) mm, half-axes 80 / 58 across, 50 / 95 high, T 17 mm, c 0.7, kv 12 mm): the side silhouette within
  3.6 mm of v2's from 1.20 to 1.36; two volumes, a fold, the upper slope easing out -- but in the level sections the most forward
  point only 44 mm out and the valley at the zip 9 mm: close-set, one broad mass from the front. Its first build also swelled the
  BACK (the field was a function of x and z alone: a notch behind the shoulder in the posed side view) and showed the radial-lift
  fold above: both fixed in the code.
- **Attempt 2** (apex (76, 1288), inwards 72 = stopping at the centre line, outwards 60, below 48, T 17.2, kv 10): the apex 49 mm
  out, the valley 15 mm, the profile kept. Kept.
- **Attempt 3**: the seam under the bust laid IN the new fold (2 mm under it, 12 points: a smooth curve under each breast meeting
  the other at the zip, as the concept draws it; B1m's six points ran 4-8 mm under the fold with a corner at each point) and the
  princess lines started from it. Kept.
- **Achieved** (the built mesh, the same measures as step 1): apex 131.3 mm at 1.278 m (v2 131.3 at 1.277; B1m 151.2 at 1.285); the
  side silhouette within 3.4 mm of v2's from 1.20 to 1.36 m (B1m up to 20 mm); 117.0 / 122.6 / 128.4 at 1.24 / 1.25 / 1.26 (v2
  114.9 / 121.6 / 126.7), 127.6 / 117.8 / 99.1 / 80.8 at 1.30 / 1.32 / 1.34 / 1.36 (v2 129.7 / 117.9 / 100.4 / 78.5); above, in the
  collar's base, it stays B1m's (64.0 / 52.7 at 1.38 / 1.40, v2 58.8 / 47.5: 5 mm, inside the registration's +-5 mm). Level sections
  at 1.27 / 1.28 / 1.29: the zip 117.0 / 116.5 / 115.0 mm in front of the centre, the breast 130.8 / 131.2 / 129.9 at x 49-50 mm: a
  valley 14-15 mm deep and broad (B1m: 125 at the zip, the breasts 147-151 at x 56-58 behind a narrow V at the zip). The body's outer
  x at 1.27-1.31 (the breasts' outer side, under the arm) 4.9-6.8 mm a side wider than B1m's; at 1.26 and below and at 1.33 and above
  within 0.5 mm.
- Sheet IoU: side above the waist 0.921 -> 0.901 (the A-pose side drawing's bust, "too much" by the owner, is the larger one), front
  above the waist 0.951 -> 0.951, back unchanged; whole side 0.905 -> 0.898.

### Step 3, the seat in profile

v2's (and v1's) arm and hand hang over the seat from the waist to below the buttock, so their back outline is not drawn. The skin
they show reaches back to the arm's front edge, so the drawn seat is AT LEAST that deep (the high-cut suit's leg line bounds the skin
above 0.99 m, so only 0.88-0.98 counts). Same landmark scaling as the bust; behind the body's centre, mm: at 0.88 / 0.91 / 0.94 / 0.96
v2's skin edge 104.7 / 111.8 / 109.9 / 106.5, B1m 87.3 / 95.4 / 98.8 / 98.5 (v1 by its own landmarks the same within 2 mm); the
fronts agree within 3-7 mm. So the seat is clearly fuller in the drawing (by 8-17 mm at least; uncertainty about 5). Moved toward it
by the seat's own two rounds only (`SEAT_B1N`: height 6.5 -> 12 mm; no profile control, no width): back outline 89.2 / 99.4 / 104.2 /
103.5 (+2.2 / +4.0 / +5.4 / +5.0 mm): half way at the crown, less on the under-curve. Hip width (front view, every 2 cm from 0.80 to
1.10) and the leg gap identical to 0.01 mm. Posed (the standing side, her right): 103.7 / 108.8 at 0.88 / 0.92 (B1m 98.4 / 103.5).
B1m's seat is kept in the shape `b1m`. Picture: `seat_profile.jpg`.

### Nothing else moved (`C:/tmp/beatrice_b1n/cmp3.py`, `cmpz.py`; `Beatrice_b1m_922fc39.blend` against this build)

27 objects identical (max |dP| < 1e-7 m): the head, hands and nails, cuffs, boots and their cords, knee pads and rims, the sleeve, the
collar, the blade insert, the zip pulls, the front zip's stop and both collar zips. Moved:
- `Suit_Body` (each new vertex's distance to B1m's surface): the front from 1.18 to 1.40 m (the bust; 0.5 mm at 1.18, 3.1 at 1.20 --
  B1m's Gaussian reached down there and is gone, and the wall eases in from 1.185 -- 9.8 at 1.22, up to 21 mm at 1.28, 3.6 at 1.38,
  0.75 at 1.40); the back from 0.80 to 1.08 (the seat, up to 5.5 mm at 0.92-0.94). Nothing else over 0.1 mm.
- `Suit_Shoulder`: its torso part beside the bust, x 0.125-0.133 at 1.26-1.30 in front, up to 6.8 mm (the breast's outer side; the
  shoulder's own surface carries the forms where the torso rules); behind the side under the arm at 1.28-1.32 up to 2.4 mm; along the
  raglan line and the shoulder insert at 1.32-1.40 (x 0.10-0.14) up to 1.9 mm. The deltoid and the arm (x > 0.146) did not move by
  0.1 mm.
- Riding on those: the front zip's tape and teeth (1.19-1.41, up to 9.3 mm), the piping (the chest seams and the seat's U), the
  shoulder insert (1.2 mm), the waist insert (0.11 mm), the thigh net (0.18 mm, the seat round's tail), the back-waist insert
  (0.05 mm), the back zip (0.001 mm).
- `source/mesh_stats.json`: 62 objects, 1,247,524 source / 3,292,804 evaluated triangles (B1m 1,247,563 / 3,292,964), all closed;
  the flipped-normal list unchanged.

### Pictures, reproduce

- `bust_profile.jpg` (v2 with B1m's and B1n's side silhouettes, and in the model's frame), `bust_b1n.jpg` (B1m above, B1n below:
  front, three-quarter, side, from below, the fold close, the concept framing, reflection lines; the concept beside both),
  `seat_profile.jpg`; `posed_vs_concept.jpg`, `posed_full.jpg`, `posed_hotlight.jpg` regenerated; the full set (sheet, turnaround,
  every beauty and dev view). Not re-rendered: `shape_vs_concept.jpg`, the `--hips` pictures, `crimson_hero.jpg`, the earlier
  rounds' before/after sheets.
- Build times: the build 5.4 min (`built in 324 s`), the full set 9.2 min; posing all shots 53 s, the hot shot 15 s.
- Reproduce: `bash charkit/run.sh Beatrice --sheet --turn --save --beauty hero,hero_back,torso,torso_back,legs,side,head,top,elevated,shoulder,shoulder_back,wrist,boot,flank,flank_back,knee,hand,boot_side,armpit_back,top_close,hand_palm`,
  then `posed.py` (all shots, and `-- --views concept --lights hot`) and `posed_sheets.py` as in B1m. The before: `--shape b1m`.
  Scratch in `C:/tmp/beatrice_b1n/`: `m_outline.py` (silhouettes and sections of a build), `prof.py` (v2's outline, the landmark
  similarity, `bust_profile.jpg`), `dump_loft.py` + `bustform.py` + `design.py` / `quick.py` / `lev.py` (the numpy design loop),
  `bh.py` + `zb_render.py` (body-only rebuild and chest close-ups: gloss, reflection lines, object ids), `seat.py`, `gapw.py`,
  `cmp3.py`, `cmpz.py`, `compose_b1n.py`.

## B2a: the head, round 1 -- the face test (2026-10-20)

The owner: "i'd like to have her face very nice ... we should try to make her face as good as we can"; "if we're gonna have
her, she must look as good as in the picture". The style (the orchestrator's, vetoable by the owner): a STYLISED anime-style
3D face, as modern anime 3D games build them. References added (the owner's, 1254 x 1254, 2026-10-19): `concept-face-front-v1.png`
(SHA-256 `c165cc1ebd223beb21f97ce06fd6a4bc0d3e70d49033ff01e03107d921e63501`) and `concept-face-side-v1.png` (SHA-256
`8b30b341610d8e20936882a542efc1a2b7b309efffeec43c13fcb565c0f983dc`). The body is frozen; the before is
`source/Beatrice_b1n_adddc04.blend` (gitignored).

### Step 1, measured and planned (before building)

- **Frames.** The bald A-pose drawings in their sheet frames (`sheet_ref.py`): front `without_hair.png` 1.1574 mm/px (centre
  column 511.5), side `without_hair_side.png` 1.1744 mm/px (y = 0 at column 518, facing image-left). The close-ups are scaled by
  the A-pose front's eye line (1.5626 m, the pupils' row 155) to chin (1.478, row 228): 84.6 mm, i.e. 0.2074 mm/px on the front
  close-up (eye row 597, midline column 653) and 0.2115 mm/px on the side close-up (eye row 567; its nose tip, column 938, put at
  y -0.104, where the A-pose forehead (-0.101 at 1.596) and the close-up's nasion 13 mm behind the tip agree).
- **The head's size (built: the A-pose drawing's).** Chin to crown: A-pose front 202 mm (rows 53.5-228), side 200 mm (rows
  36.5-207); built 1.476 to 1.680 = 204 mm (the placeholder's mesh: 183). Cranium 150 mm wide at 1.60 (A-pose front, ears
  excluded) and 184 mm deep (side: -0.101 to +0.083). Cheeks 124-128 mm wide at 1.545 (the front close-up: the locks over its
  cheeks hide the edge, read by eye; the A-pose 135). Neck: drawn 70-73 mm wide just above the collar (both); built 76 mm at
  1.50, but the placeholder's rings kept up to the collar's top (1.474-1.482: 84 mm), because the collar was flared round them
  and rolls in onto the neck -- the collar is NOT changed.
- **The concept's ratio differs by more than 5 %.** On `posed_vs_concept.jpg`'s framing (1482 px/m at the figure; the shoulder
  line and the crotch at the concept's heights): eye-to-chin concept 158 px against the model's 125 (x 1.26, against the torso's
  length); against the shoulders with the arms (630 / 543 px) x 1.09; the cheeks' width x 1.30. So a named variant with the head
  scaled x 1.20 about the chin's height on the neck's axis (the neck itself unchanged in the collar): `--head concept`.
- **The face, front close-up** (mm from the midline / the eye line, her left eye): pupils 70.7 apart (the A-pose 73); the eye
  opening from the inner corner (16.6, -3.9) to the outer (57.4, +1.9) -- 41 mm wide, tilted up 8 deg outwards -- 18.5 mm high
  (upper lid's top +8.5, lower lid's bottom -10.4); the iris 19.5 mm across, its centre 33.4 out, its top under the upper lid,
  its bottom touching the lower one; the pupil 5 mm; the big highlight up and outwards of the pupil, a small one beside it.
  Brows 17 mm above the pupils, mostly under the fringe. Nose tip 26 mm under the eye line (both close-ups), the nostrils 36
  (-> 1.5263), 6 mm apart. Mouth 26 mm wide at 55 mm under the eye line (1.5074; both close-ups and the A-pose agree). The jaw
  (the drawn edge against the shadowed neck): half-widths 51.5 / 47.6 / 43.5 / 39.5 / 34.9 / 29.8 / 24.2 / 18.3 / 12.0 / 5.0 mm at
  1.5226 / 1.514 / 1.508 / 1.504 / 1.500 / 1.4956 / 1.4915 / 1.487 / 1.483 / 1.479: a narrow, softly pointed V. Ears 1.577-1.535.
- **The profile, side close-up** (mm behind the nose tip at 1.5366): nasion 13.1 at 1.547, the bridge straight up to the tip, the
  underside sloping back to the subnasale (9 behind at 1.52: upturned), the upper lip 9.9 (1.513-1.518), the mouth line 16.1
  (1.507), the lower lip 15.7 (1.505), the chin receding smoothly 20.5 / 23.7 / 25.6 at 1.4964 / 1.4879 / 1.4795, its underside
  at 1.476 running back and up about 11 deg to the neck. The ear 1.575-1.528, its front edge 105 mm behind the tip.
- **Where the two disagree, and what was done.** The A-pose side has almost no nose (its tip level with the forehead); the
  close-up's projects 15 mm: the profile rules depths, so the close-up's nose, lips and chin. The A-pose's face is wider (cheeks
  135 against about 126 mm) and its ear 18 mm further back (124 mm behind the nose tip against 106): the front close-up rules
  widths (the face built 126 mm), the ear placed between (front edge 2 mm behind the head's centre, about 110 mm behind the tip).
  The skull outside the face (crown, back, cranium width) is the A-pose's, which alone draws it; the hair hides it.
- **The plan.** One implicit surface for skull, face and neck (the B1h-B1n recipe: C2 unions, a banded mesh projected onto the
  field, the field's own normals; zebra proof). Eyes: NOT eyeballs (a 41 mm almond would need a 22 mm ball); the opening is carved
  through the skin over the drawn almond (front projection), its margin rounded C2 into a lid; behind it an eye SURFACE that is the
  skin's own surface set back 1.2 mm (deeper to the corners: a gentle dome), on which the iris, pupil, lid shadow and highlights are
  PAINTED along the view axis (attributes `iu`, `iv`, `lid`), so the front reads exactly as drawn and the iris foreshortens like a
  drawing on a curved face as the head turns; the heavy upper lash line a swept dark band with an outer flick (geometry: it stands
  off the lid in three-quarter and profile); the lower lash line, the lid crease and the brows painted. Nose and mouth: small forms
  (shape) with the drawing's tip highlight and the mouth line carried by shading and paint. Skin: the hands' base, subsurface and
  gloss without pores; blush, lip tint, mouth line, lash line, nostrils as `PALETTE` entries. Hair: a blockout -- a cap following the
  skull, diving under the skin below the hairline, and smooth locks (fringe, crown, the locks in front of the ears, over the
  shoulders, the mass down the back) as swept ribbons, the hanging parts dropped as ropes kept clear of the head and the suit.
