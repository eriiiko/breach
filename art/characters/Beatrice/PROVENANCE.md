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

### Step 2, the face

- **Mechanism** (`charkit/animehead.py`, new; a later anime-style character is a set of tables on it). The skin is ONE field
  (head space = the tables' frame): an `Egg` -- per elevation about the centre (0, -0.005, 1.575) the profile's front and back
  radii and the front view's side radius, each a natural cubic through the hand-picked outline points of step 1 converted to
  polar, blended round the head by superellipse exponents (front 2.6 across the eyes, 1.8 at the chin: the V) -- minus Gaussian
  forms (socket -2.0 mm, brow +1.2, cheeks +2.0, chin +1.2, the dent under the lip -0.8), C2-united (`smin3`) with the nose
  (bridge capsule, tip and column, small alae) and the lips (two ellipsoids), cut by the mouth line (a 0.9 mm slot) and the eye
  openings (the drawn lid curves in the front view, a prism along the view axis, only within 5 mm of the egg, its margin rounded
  C2 over 1.2 mm: the lid), C2-united with the neck (`Rings`; the placeholder's rings up to the collar's top, slimmer above).
  Meshed by surface nets on a banded grid (only blocks near the surface sampled finely; memory scales with the surface), every
  vertex projected onto the field, shaded with the field's normals -- except the nose's bridge and sides and the lips, which are
  shaded with the normals of the face WITHOUT them (`shading`): the anime-3D trick that leaves the profile's nose and shows the
  front only its tip. Painted masks per vertex from the front-projected position (`blush`, `lips`, `mouthline`, `lowlash`,
  `crease`, `brow`, `nostril`, `nosetip`) read by the new `beatrice_face` material (the hands' skin base, subsurface and gloss, no
  pores). The head-size variant is the same tables scaled about the chin's height on the neck's axis with the neck left in the
  collar (`Head(scale=)`).
- **The eyes.** Not eyeballs: an eye SURFACE per side, the uncut skin's own surface set 1.2 mm back behind the opening (2.5 mm
  more towards the corners: a gentle dome), kept only over the opening and a 1.8 mm margin, 0.8 mm thick; on it the drawing is
  PAINTED along the view axis (`iu`, `iv`, `lid` attributes; material `beatrice_eyes`): sclera shaded under the upper lid, an iris
  10.0 x 11.2 mm half-axes, dark at the top, a lit glow at the bottom, radial streaks, a dark rim, a 3 x 3.6 mm pupil, the two
  highlights (emissive) up and outwards. The heavy upper lash line is geometry (`beatrice_lash`): a dark band swept along the upper
  lid on the skin, 1 mm at the inner corner widening to 4.2 mm over the outer third, its lower edge 1.3 mm over the opening (it
  covers the iris's top as drawn), ending in a flick 6.5 mm past the outer corner. The lower lash line, the lid crease and the
  brows are painted. Turning: from the front the eye is the drawing (the iris's top under the band, its bottom on the lower
  lid); in three-quarter the painted iris foreshortens with the face's curve and the band stands off the lid with its flick; in
  profile the eye shows as the drawn convention does -- a narrow sliver of iris behind the band and flick -- because the eye
  surface follows the face's curvature (`face_turn.jpg`, `face_closeup.jpg`).
- **The ears** (`animehead.Ear`): a field each in its own axes (tilted back 12 deg, flared 20 deg): two plates (the auricle and
  the lobe), the bowl and the concha taken from the outer face leaving the helix, an antihelix and a tragus; 47 x 27 mm, top
  1.577, lobe 1.53, as both close-ups draw them.
- **The three worst defects after the first full build, two attempts each** (scratch renders `C:/tmp/beatrice_b2a/o_a0..a7`):
  1. *The nose read long and realistic from the front* (a lit ridge and nostril wings; the drawing: only a tip highlight).
     Attempt 1: the bridge narrowed (3.2 -> 2.4 mm), the tip smaller, the alae shrunk and pulled back, a wider fillet: still a
     ridge with nostril shading. Attempt 2: the bridge and sides shaded with the face-without-nose normals (tip kept): from the
     front a soft shadow down one side and the tip, the profile unchanged. Better; in the close-up it still reads as a smudge
     rather than a crisp drawn shadow.
  2. *The eyes read wide-open, startled* (a thin lid line, the iris floating in white). Attempt 1: the iris larger and lower, the
     lash band wider and lowered over the opening. Attempt 2: the band heavier still and further over the iris, the flick longer.
     Now the drawing's heavy upper line; still missing the lash band's broken upper edge (separate lash clumps) and the drawing's
     soft lid shadow on the iris.
  3. *The ears*: the shared lofted ear (`parts.ear`) came out pointed, an elf's, in profile. Attempt 1: an ear as a field -- the
     bowl cut through the too-thin plates and left a hollow ring. Attempt 2: plates 12 mm thick at the centre, a shallower bowl: a
     solid small ear of the right size and place; simple (no helix fold or antihelix Y worth the name).

### Step 3, the hair blockout

`charkit/animehair.py` (new), table `HAIR`: a cap -- the egg offset outwards 6 / 12 / 9 mm (front / crown / back) above a
hairline (alpha, z) and 9 mm inside the head below it, so the visible hairline is a clean intersection (above the ears, the
ears exposed; the nape at 1.515) -- and 50 locks in six mirrored rows, each a closed swept ribbon (lens section, tapering to a
point) laid on the cap along a path in the head's angles: the fringe (12, parted at the centre, the middle tips at 1.567 between
the eyes, the outer ones higher, standing 4 mm off the forehead), the crown (10), the locks in front of the ears (6, down in front
of the shoulders), over the shoulders from behind the ears (4), the outer back (8, leaving the skull at its widest), the back (10,
to the V's point at about 1.20). The hanging parts fall as ropes under gravity (position-based: lengths, a bending limit, pinned
roots) kept 4-6 mm off the cap and off the suit's OUTER surfaces (a BVH of Suit_Body, Shoulder, Collar, Sleeve, the blade and
shoulder inserts with their Solidify muted -- with it, a rope that started inside the body stayed inside against the inner wall).
Material `beatrice_hair`: dark brown, streaks across each lock (`hv`), roughness 0.42, a little coat and sheen. In the posing the
hair is attached wholly to the head bone (`POSE["attach"]`, new in `suitpose.deform`).

### Pictures, checks, reproduce

- `face_vs_art.jpg` (the close-ups and the model in their own frames, orthographic, with hair and without: drawing | render | the
  render's outline over the drawing, front and profile), `face_turn.jpg` (0 / 30 / 60 / 90 / 135 / 180 deg, the optional
  `portrait` light: a soft key above her right, a fill, a rim; the default studio rig is unchanged), `face_closeup.jpg` (eyes, an
  eye in three-quarter, nose and mouth, the mouth from her right, beside crops of the drawings), `face_zebra.jpg` (reflection
  lines on skin and ears: front, three-quarter, side -- smooth bands, no rings or seams; the bands bend gently over the cheek's and
  socket's Gaussian forms and are busier round the nose and mouth), `head_size.jpg` (the built head against the concept-ratio
  variant, front and 30 deg, beside the concept); `posed_vs_concept.jpg`, `posed_full.jpg`, `posed_hotlight.jpg` regenerated (the
  head joins the posing by the existing weight field; the hair attached wholly to the head bone); the full standard set (sheet,
  turnaround, every beauty and dev view).
- **Sheet scores** move because the drawings are bald and the model now has hair: front above the waist 0.951 -> 0.889, side
  0.901 -> 0.829, back 0.918 -> 0.869; the bands below the waist unchanged.
- **Nothing else moved** (`C:/tmp/beatrice_b1n/cmp3.py`, `Beatrice_b1n_adddc04.blend` against this build): 35 objects identical
  (max |dP| < 1e-7 m) -- every suit part including the collar, the hands and nails, the boots. Changed: the `Head` collection only
  (`Beatrice_Head`, `_Ear`; new `_Eye`, `_Lash`, `_Hair_Cap`, `_Hair_Locks`; gone the placeholder's `_Eye_L/R` and their lids).
  The neck inside the collar keeps the placeholder's rings up to 1.474 (and the collar's hug at 1.482), so the collar did not move.
- `source/mesh_stats.json`: 62 objects (the Head collection 6, as before), all closed, the flipped-normal list unchanged; the
  head 1.16 M source triangles (the skin at 0.7 mm: 384 k vertices).
- Reproduce (from `art/characters`): the build and the standard set as in B1n (`bash charkit/run.sh Beatrice --sheet --turn --save
  --beauty ...`); `--head placeholder` builds the bald placeholder, `--head concept` the x 1.20 variant; the face pictures:

      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/face_views.py
      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/face_views.py -- --head concept --only turn
      python Beatrice/scripts/face_sheets.py

  then `posed.py` (all shots, and `-- --views concept --lights hot`) and `posed_sheets.py` as in B1m.
- Times: the build 7.8-9.3 min (the head 95 s of it: the skin at 0.7 mm; the hair 11 s), the full set 11.8-13.5 min (another Blender process shared the machine during the first); the face views
  1.5 min, the variant 3 min (its head rebuilt), posing 1.1 min + 0.3. The design loop: `C:/tmp/beatrice_b2a/tools/design.py`
  (numpy: the profile and silhouettes against the drawings, 30 s), `hb.py` (the head rebuilt on the B1n build and rendered, about
  1.5 min), `hairgap.py` (the hair's signed gap to the suit's outer surface).
- **The hair against the suit** (`hairgap.py`, the suit's outer surfaces incl. piping and zips): the ropes keep each lock's centre
  4-6 mm off; a ribbon's edges dipped up to 3.7 mm into the shoulders' curve (599 vertices), so every lock vertex below the head
  is lifted 1.2 mm clear along the suit's normal (1338 vertices moved): now 14 vertices touch (-0.06 mm, on piping), none inside
  the suit; the cap 0.9 mm off at the nearest. Unposed only; posed, the hair is rigid with the head and not re-checked. The back's
  tips reach 1.08-1.25 m (the V's point lower than the drawings' 1.17-1.23: the ropes stretch as they settle).

### What still looks wrong (worst first)

1. **The face reads as a doll, not as the drawing** (`face_vs_art.jpg`, `face_turn.jpg`): the drawing's look is mostly PAINTING --
   soft shadow under the fringe and lids, a warm gradient, inked lid lines, translucent iris layers, glossy lip highlight -- and
   the render has flat, bright, evenly lit skin. Needs a toon/anime shading layer (shadow ramp, lid and fringe shadows painted or
   ramped, rim tint) rather than more geometry.
2. **The nose** (`face_closeup.jpg`): from the front a smudge down one side instead of the drawn tip highlight; in three-quarter
   a dark diamond between the tip, column and alae. The normal-transfer route is right; it needs a cleaner proxy (a smooth front
   plane), the alae and column removed from the geometry, and the nostrils and tip highlight painted.
3. **The eyes** are the strongest part but still graphic-flat: the lash band is a solid tube with no lash clumps on its upper edge
   and no lower-lid thickness; the iris lacks the drawing's layered glow; the lower lash line is a thin painted line; the eyes sit
   a touch too round and wide-open compared with the drawing's narrower, heavy-lidded look (`face_closeup.jpg`, `eyes`).
4. **The hair** is a blockout: thick slab ribbons with visible edges, a comb-like fringe of identical spikes, a seam where the
   fringe's roots meet the crown locks, a narrow back curtain (the drawing's hair spreads wider over the shoulders and frames the
   face with loose strands); too glossy-plastic (`face_turn.jpg` 135 / 180 deg).
5. **The mouth** (`face_closeup.jpg`): two pink bands with a groove; the drawing's soft upper-lip line and glossy lower lip need
   paint and a gentler form. The profile's lips sit about right.
6. **The ears** are simple and pale (no helix fold, no inner shading; `face_turn.jpg` 90 deg).
7. **The head's size**: built at the A-pose drawing's 204 mm, it reads small against the concept (`posed_vs_concept.jpg`); the x 1.20
   variant (`head_size.jpg`) is much closer to the concept's feel -- a choice for the owner.

## B2b: the anime look, the face's form and the eyes (2026-10-20)

The owner's decisions of 2026-10-20 (not reopened): an anime / toon look for the WHOLE character in her high-resolution renders
(promo, cutscenes, our pictures; the game's top-down renderer untouched); the x 1.20 head (the concept's ratio) as the default,
the A-pose size kept as a named variant; on the B2a face: "some more work on the eyes angle ... and the facial form too ... it's
getting pretty close". The before is `source/Beatrice_b2a_0d9d90f.blend` (gitignored).

### Step 1, the x 1.20 head by default

`beatrice.HEAD` is the anime head at `HEAD_SCALE = 1.20` (scaled about `animehead.Head`'s pivot, the chin's height on the neck's
axis; the neck stays in the collar); `HEADS`: `anime` = `concept` = the default, `apose` = B2a's head (204 mm chin to crown),
`placeholder` the bald one. The skin's meshing box now grows with the scale (`Head.box`): B2a's fixed box would have cut the
x 1.20 crown flat at 1.692 m and clipped the nose's tip by 0.8 mm. The face pictures' frames (`FACE_SHOTS`) are in head space
and are placed and sized by the scale (`face_views.py`), so the close-up overlays still lie on the drawings.
- **The neck and the collar**: the neck rings are not scaled, so nothing in or at the collar moved (vertex check below: the
  collar identical); at x 1.20 the neck reads slimmer under the larger jaw, as the concept draws it (`face_turn.jpg`,
  `posed_vs_concept.jpg`). No change to the collar's neck edge was needed.
- **The hair against the suit** (B2a's `hairgap.py`, the suit's outer surfaces incl. piping and zip): the locks' nearest points
  touch the piping (6 vertices at -0.02 mm, B2a: 14 at -0.06), none inside the suit; the cap's lowest ring now reaches 2.4 mm
  behind the collar's outer surface at the nape (20 vertices; B2a: 0.9 mm clear) -- inside the neck and the collar, hidden in
  every picture (`hero_back.jpg`, `torso_back.jpg`, `face_turn.jpg` 135 / 180 deg).
- **Sheet scores** fall with the larger head against the bald A-pose drawings: above the waist front 0.889 -> 0.837, side
  0.829 -> 0.747, back 0.869 -> 0.817; the bands below the waist unchanged. (The drawings are the A-pose size by design; the
  concept is the reference for the ratio.)

### Step 2, the face's form and the eyes (measured first)

Measured on the close-ups in B2a's frames (`concept-face-front-v1.png` 0.2074 mm/px, eye line 1.5626 at row 597;
`concept-face-side-v1.png` 0.2115 mm/px), head space (the A-pose size; the default is the same shape x 1.20). Tools:
`C:/tmp/beatrice_b2b/tools/eyegrid.py` / `eyecheck.py` (the drawn eye on a 1 mm grid with the tables' curves over it),
`profile_cmp.py` (the drawn facial profile, the right-most skin pixel per row, against the field's mid-plane profile),
`eggcurv.py` (the egg's curvature along the elevation), `jaw.py` (the front's occluding contour), `nosehole.py`.

- **The eyes' angle and shape** (her left eye, mm from the midline / the eye line):

  | | B2a | drawn | B2b |
  |---|---|---|---|
  | opening, inner -> outer | 16.6 -> 59.0 (42 mm) | 18 -> 52.5 (35 mm) | 18.0 -> 52.4 (34.4 mm) |
  | inner corner / outer end | -3.9 / +0.4, a raised point | -2.6 / -2 .. +2, round and blunt | -2.6 / +0.2, round |
  | corner-to-corner slope | about 6 deg up (8 by B2a's points) | about 4-5 deg | 4.7 deg |
  | upper lid's top | +8.5 at 33-41, then a long slope to the point | +7.2 .. +7.4 held to 42, then a round drop | +7.4 held to 41, then the drop |
  | lower lid's bottom | -10.0 | -9.7 at 36 | -9.6 at 35 |
  | lash flick | to 65 mm, 6 mm over the corner (a long wing) | ends about 57 mm, nearly level | to 56.8 mm at +5.6 |
  | iris centre / radii | 33.4, -0.4 / 10.0 x 11.2 (its bottom 2 mm under the lid) | 33.7, +0.2 / 9.7 x 10.0 (bottom on the lid) | 33.7, +0.2 / 9.7 x 10.0 |
  | pupil | 6.0 x 7.2 | 4.9 x 6.2 | 4.9 x 6.2 (`iris["pupil"]`) |
  | gaze | straight | straight at the viewer | straight |

  The cat-eye of B2a was the long pointed outer corner and the long raised wing; the drawn eye is a rounder, heavier, nearly
  level arch with a blunt outer end and a short flick. The lash band follows the new lid to 50.5 mm, then a short flick; it covers
  the opening's top by 0.6 mm (B2a 1.3: the lid is now traced on the drawn one). The lower lash line, crease and lid shadow are
  stronger, the sclera whiter, the iris deeper with a lighter ring round the pupil and a soft glow low on the far side (palette
  and `animehead.materials`). **Three-quarter**: the face is rounder across the eyes (`n_front` 2.6 -> 2.35 at the eyes) and the
  eye surface's dome deeper towards the corners (`curve` 2.5 -> 3.5 mm), so the eyes sit on a face that curves away
  (`face_closeup.jpg`, three-quarter) rather than on a flat mask. Attempts: one on the eye tables (the traced shape held);
  the paint and colours were tuned twice in the toon look.
- **The face's form**:
  - *The nose's root* (profile): B2a's stood 4-6 mm in front of the drawn one between 1.549 and 1.564, so its bridge read short and
    scooped. A midline dent (a Gaussian, -6.5 mm at 1.5585) and the bridge's root moved 4 mm back: now within 2.6 mm (1.5514
    -2.5, 1.5535 -1.9, 1.5556 -1.0, 1.5577 -0.4 mm; + = model behind); tip, lips and chin unchanged (within 1.5 mm, as B2a).
    Attempts: 1 (the dent at -4.2 mm, the root 1.5 mm back: 2-3.5 mm forward still), 2 (kept).
  - *A tunnel through the nose*: B2a's "dark diamond" under the bridge in profile was the BACKGROUND seen through the nose -- the
    bridge, tip and column made an arch with nothing under it once the alae were shrunk to nothing (144 uncovered cells of 0.5 mm
    between face and nose, 1.529-1.540). A filler on the midline under the bridge (`nose["alae"]`) closes it (0 cells); the profile
    is unchanged.
  - *Bands across the cheeks*: the egg used the PROFILE's front radius at every azimuth, so the profile's own dents (under the nose,
    the lip, the chin) printed as concave horizontal bands across the cheeks (curvature along the elevation at 45 deg round:
    -5 and -7 /m at -45 and -25 deg) -- the stripes on B2a's profile cheek. Now away from the midline the front radius is a smoothed
    copy (`egg["front_smooth"]` = 12 deg along the elevation, blended in over 14 deg round the head; the midline keeps the drawn
    profile exactly): no concavity below -2 /m off the midline.
  - *The jaw's curve into the chin* (front, half-widths drawn / B2a / B2b, mm): 1.5226 51.5 / 50.8 / 50.8; 1.514 47.6 / 46.1 / 45.7;
    1.504 39.5 / 38.3 / 37.5; 1.4956 29.8 / 31.1 / 28.4; 1.4915 24.2 / 27.4 / 21.8; 1.487 18.3 / 22.7 / 15.5; 1.483 12.0 / 17.9 / 11.4;
    1.479 5.0 / 11.6 / 7.5 -- B2a's chin was blunt by 5-6 mm, B2b's within 3 mm everywhere (a by-product of the smoothing).
  - *Unchanged, measured*: the cheeks 123 mm wide at 1.545 (the front close-up 124-128 behind the locks; the A-pose 135); the
    midface (the nose tip 26 mm under the eye line, as drawn), the mouth (55 mm under, as drawn), the lips' profile (within 1.3 mm).
  - *From the front*, the nose as a long dark ridge was SHADING, not form: the toon look shades the face as a simple head and
    leaves the nose's lines out when seen from the front (step 3), so it now reads as the drawn tip highlight and two nostril marks.

### Step 3, the anime look

**Technique (chosen for quality first): EEVEE (Blender 4.5) with Shader-to-RGB**, render-time only (`charkit/toonlook.py`, the
mechanism, character-free; `charkit/suitlook.py`, her tables to it; tables `LOOKS` / `LOOK` and the new `PALETTE` entries in
`beatrice.py`). Why: the look needs the light as a number a material can ramp (a two- or three-tone terminator, a highlight that is
either there or not) AND real cast shadows (the fringe on the forehead, the chin on the neck); Shader-to-RGB gives exactly that,
Cycles cannot (a ramp on a hand-made N.L has no shadows). Lines: an **inverted hull** per object (a flipped Solidify shell,
back-face culled, invisible to shadows, its colour per material from the palette, its width set per shot in PIXELS at the
subject); Freestyle and Line Art cost minutes a frame on 4.8 M evaluated triangles and draw noisy creases on surface-net meshes.
The hull draws the silhouette and every overlap (the chin on the neck, the hair on the face, an arm over the body), as the art
inks them. The B1a-B2a look is `--look pbr` everywhere (Cycles, unchanged).

- **Materials** (each rebuilt round what fed its Principled BSDF's Base Color and Normal, so the painted masks, nets and piping
  stay): L = Shader-to-RGB of a white Diffuse (1 = square to the key), colour = own colour x a `ramp` over L (EASE), + highlight
  layers (Shader-to-RGB of a Glossy through `lo`..`hi`: crisp), + a facing `rim`, + the hair's band; emitted as is. Ambient
  occlusion in the base colours is switched off (it printed soft dirt gradients).
  - *Skin*: a warm three-tone ramp (shadow a warm tan, a saturated warm terminator, lit); the hands' ramp sharper, the face's
    lit nearly whole (its shadow only at the far edge). **The face is shaded as a simple head** (`Head.toon_normals`): custom normals
    blended to the egg's (no forms, nose, lips or cuts), tipped forward (`bias`) and, over the jaw's side, up (`lift`, so the jaw
    reads lit to its edge over the shadowed neck); **and it casts as one** (`Head.shadow_proxy`: the egg and neck shrunk 9 mm, a
    shadow-only mesh; the skin and the hair's cap do not cast): no nose, lid or socket shadow falls on the face, the fringe's and
    the chin's still do. **Lines left out from the front** (`Head.line_mask`, `toonlook.line_group`): the hull's width over the
    nose and lips scales with how frontal the view is, so the front shows the drawn tip and the mouth line only and the profile
    keeps the nose's outline.
  - *Eyes*: painted, barely lit (`flat` 0.85): the layered iris (dark top and rim, a lighter ring round the pupil, the glow low on
    the far side, the two highlights), the lid's shadow; the lash band flat dark.
  - *Hair*: dark brown, a soft two-tone ramp, normals blended towards the head's centre above the jaw (`toonlook.radial_normals`)
    so it shades as one mass and the slabs' edges do not each carry a terminator; the "angel ring" a jagged band of elevation round
    the head (`toonlook.ring_attr`, 42 deg, `HAIR_RING`) on the lit side.
  - *Suit, boots, pads* (`_LATEX`): a near-black base lifted where lit, a broad soft grey sheen (Glossy 0.30), crisp white streaks
    (Glossy 0.08 through 0.95..1.10) thrown by narrow strip lights that light nothing diffusely, a cool rim; the piping's relief kept
    (bump x 0.5). Inserts: the net over the skin, ramped; the zip metal ramped with a highlight.
- **The light rig** (`LOOK_TOON["lights"]`, parented to the studio rig so a back view is lit as a front view): a sun key above her
  right front (42 deg up, 20 deg round), a rim sun behind her left (diffuse x 0.35), three strip / small area lights with no
  diffuse part for the latex's streaks; a flat ambient world, the backdrop for camera rays; a toon floor (the backdrop, darkened
  under the key's shadow; alpha = the shadow in transparent renders).
- **Where it is used**: the default for the turnaround, beauty and dev views (`run.sh ... --look NAME`, default `LOOK`), the face
  pictures (`face_views.py`), the posing (`posed.py`). The sheet's overlap pictures are rendered plain (Cycles) before the look is
  applied, so their scores compare with B2a's. Raw renders of a non-default look get a `<look>_` prefix.
- **Cost**: EEVEE, 64 samples, RTX 3070: a face shot (720 x 900 to 1254 x 1254) 3-4 s, a posed shot (1024 x 1536) 6-8 s, the
  first shot of a process 10-40 s more (shader compilation); applying the look 7-13 s (the face's normals and the proxy from the
  fields). The same posed set in pbr (Cycles, 96 samples) about 11-12 s a shot. The build is unchanged (8.1 min); the full set
  with every beauty and dev view 11.3 min.

### Pictures, checks, reproduce

- `look_compare.jpg` (pbr | toon: face front, 30 deg, the posed concept framing, the posed full figure), `face_vs_art.jpg` (toon
  with and without hair, one pbr row), `face_turn.jpg` (toon 0-180 deg, a pbr row at 0 / 30 / 90), `face_closeup.jpg` (toon, the
  drawing's crops, the pbr eyes), `face_zebra.jpg` (reflection lines on the B2b skin and ears: front, three-quarter, side),
  `head_size.jpg` (the default against `apose`, each framed on its own head, so the BODY's size shows the ratio), `posed_vs_concept.jpg`, `posed_full.jpg` (toon), `posed_hotlight.jpg` (pbr studio,
  pbr hot, toon); the full standard set (sheet, turnaround, every beauty and dev view) -- the sheet plain, everything else toon.
- **Nothing else moved** (`C:/tmp/beatrice_b2b/tools/cmp.py`, B2a's `cmp3.py` with the Head collection listed, not compared;
  `Beatrice_b2a_0d9d90f.blend` against this build): 35 objects identical (max |dP| < 1e-7 m) -- every suit part including the
  collar, the hands and nails, the boots. Changed: the Head collection only (skin 383,694 -> 518,386 vertices at x 1.20, eye, hair).
- `source/mesh_stats.json`: 62 objects, all closed.
- Reproduce (from `art/characters`):

      bash charkit/run.sh Beatrice --sheet --turn --save --beauty hero,hero_back,torso,torso_back,legs,side,head,top,elevated,shoulder,shoulder_back,wrist,boot,flank,flank_back,knee,hand,boot_side,armpit_back,top_close,hand_palm
      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/face_views.py
      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/face_views.py -- --look pbr
      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/face_views.py -- --head apose --only turn
      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/posed.py
      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/posed.py -- --look pbr
      "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/posed.py -- --look pbr --views concept --lights hot
      python Beatrice/scripts/posed_sheets.py
      python Beatrice/scripts/face_sheets.py

  The head loop: `C:/tmp/beatrice_b2b/tools/hb2.py` (the head rebuilt on the B2a build, the look applied, close-ups; 2.5 min, 35 s
  with `--keep` on a saved head).

### What still looks wrong (worst first)

1. **The latex's highlights in close shots** (`face_turn.jpg` 0-60 deg, `head.jpg`, `look_compare.jpg` face cells): the key strip
   reflects in the nearly flat upper chest as large, ragged white blobs; at full figure (`posed_full.jpg`, `hero.jpg`) the same
   streaks read well. Needs the streak threshold and the strips' sizes per shot distance (or a narrower highlight core), and the
   collar's own reflections.
2. **The far cheek in three-quarter** (`face_turn.jpg` 30 and 60 deg, `face_closeup.jpg` three-quarter): the face's warm terminator
   band, then the rim's light at the edge, read as a dent running from the eye to the jaw. A softer face ramp, or a face that
   ignores the rim sun, would take it out.
3. **The hair** is still B2a's blockout (as briefed): a comb of identical spikes for the fringe, flat slab ends at the side locks
   (`face_vs_art.jpg` profile), a bald crown seam at the parting, a narrow back curtain; the toon look hides the slabs' edges
   (the radial normals) and gives the drawn highlight band, but the band is one smooth wave, not the drawn broken highlights.
4. **The mouth** (`face_closeup.jpg`): the upper lip's tint still reads as a pale lozenge above a hard line; the drawing's soft
   upper-lip line and glossy lower lip are not there (paint only; no lower-lip highlight yet).
5. **The nose tip** reads as a round pale dot (`face_closeup.jpg`); the drawing's highlight is a larger soft triangle with two
   curved nostril strokes (ours: two dots).
6. **The eyes**: close to the drawing in shape and angle now; still missing the lash band's broken upper edge (lash clumps), the
   drawing's lower lid thickness and its soft pink inner corner; the iris's layers are simpler than drawn (`face_closeup.jpg`).
7. **Sheet scores** above the waist are lower with the larger head (above); expected, the drawings are at the A-pose size.
8. **The head's interior**: the hair cap's lowest ring passes 2.4 mm behind the collar's surface at the nape (hidden).

## B2c: the hair (2026-10-20, answers `previews/review_stage11.md` section 6 and items 1, 7, 9)

The before is `source/Beatrice_b2b_3292a4c.blend` (gitignored). Only the `Head` collection's hair objects change.

### Step 1, the art read (traced before the build; this record written right after the first build's commit)

Tools: `C:/tmp/beatrice_b2c/tools/trace.py` (the hair's outer silhouette per row on the close-ups: pixels differing from the
backdrop by more than 45 summed RGB) and `trace2.py` (the same on the A-pose drawings), crops on a 10 mm grid
(`C:/tmp/beatrice_b2c/v/fringe_grid.jpg`). Head space (A-pose units, mm; the default head is x 1.20), the close-ups in B2a's
frames (front 0.2074 mm/px, eye line 1.5626 at row 597; profile 0.2115 mm/px).

- **Outer silhouette, front close-up** (her right / her left edge from the midline, width): crown 1.676: -72 / +66 (138);
  1.6449: -90 / +82 (172); 1.6035: -106 / +96 (202); **eye line 1.5626: -116 / +107 (222)**; 1.5205: -128 / +111 (239);
  **chin 1.478: -132 / +114 (246)**, the left edge then clipped by the frame; 1.4375: at least 254. The hair stands 26 mm off
  the skull at the temples (the skull's half-width 75 at 1.60), 44 mm at the eye line; the drawing is a little wider on her
  right. The crown runs out of the frame at the top (above 1.686); the A-pose drawing `with_hair.png` agrees on the widths
  (1.64: 186, 1.60: 222, 1.563: 235) and draws the crown about 15-20 mm over the skull.
- **Profile close-up**: the fringe stands well off the face -- its front 120 mm in front of the head's centre at 1.598, where
  the forehead is at 101: about 17-20 mm off at the brows, the tips hanging in front of the eye (-112 at the eye line); the back
  is out of the frame behind 94 mm. **The ear shows whole** in profile: one narrow lock passes in front of it (between the eye
  and the ear, from the temple down past the jaw), the mass behind starts right behind the ear and falls straight down.
- **The part**: a little to her right of the midline (about -4 mm at the crown's front, a pale whorl), the fringe sweeping from it.
- **The fringe** (tips, x / z above the eye line, mm): an outer clump over her right brow's outer end (x -55..-36, tips at
  +16..+18); a **gap of forehead** at x -36..-23 from +51 down to the brow; the central mass x -23..+19, five clumps with pointed
  tips at (-19, +11), (-11, +5), (**-2, +1: the middle one to the lash line between the eyes**), (+7, +6), (+15, +12); a second
  gap at x +19..+32 (+52 to +20); an outer clump over her left brow (x +32..+50, tips at +22..+26). The outer clumps blend into
  the side locks.
- **Side locks**: from the temples, in front of the ears, close to the cheek, over the jaw's corners and on past the chin onto
  the collar and chest; in the front view they hide most of her right ear and the front of her left one (her left ear shows at
  the face's edge, against the darker mass behind).
- **Loose strands**: a few thin strands crossing the cheeks' outer edges (her left: from the side lock in towards the mouth's
  corner height), and flyaways from the outer mass.
- **The mass and the back** (`with_hair*.png`, `concept.jfif`, `concept-upper-body-side-v2.png`): wider than the head from the
  cheeks down, flaring onto the shoulders, locks in front of both shoulders onto the chest; in profile the hair falls behind the
  shoulder; the back is widest at the shoulder blades and ends in a V of separate, slightly wavy, tapered tips at about mid-back
  (B2a measured 1.17-1.23 m).

### Step 2, the build


- **Mechanism** (`charkit/animehair.py`, new `Clumps` / `build_clumps`, chosen by `kind="clumps"`; B2a's blockout path is untouched
  and builds as `--head hair_b2a`). Each lock is ONE row entry (per-lock lists in `HAIR_B2C`, `beatrice.py`), not a row spread by
  rule: an on-head part laid over the cap from a root (beta, alpha) through an optional `mid` to a `leave` point -- or, for the
  fringe, to a front-view `tip` (x, z) traced on the close-up and found on the face by a ray from the front -- its offset off the
  cap growing from `off[0]` to `off[1]` plus a `bulge` (the crown's volume), the root sunk into the cap; then a FALLING part on a
  traced ENVELOPE (`envelope`: half-width, front and back from the head's axis per height, a superellipse round the head): from
  the leave point down to `z` at azimuth `alpha`, its distance from the axis `r_skin + spread x (r_env - r_skin)` (`spread` keyed
  along the fall: 0 lies on the cheek, 1 on the outer silhouette), plus a flare at the tips (`add`), a soft S round the head
  (`wave`) and a ragged length (`z_jit`). The path is pushed off the skin, the ears and the cap (fields) and off the suit's outer
  surfaces (B2a's BVH), smoothed and re-pushed, resampled every 1.8 mm. The section is a lens (`width` x `thick` along the whole
  lock, every width profile tapered to a point) whose edges curl towards what it lies on (`curl` x half-width), so a clump reads
  round with tucked edges, never a slab. Attributes per vertex: `hv`/`hu` (streaks), `hw` (across, -1..1), `ht` (along),
  `ring_off` (the lock's own highlight shift), `hshade` (the layer's darkening).
- **The layout** (48 locks, head space, the default head x 1.20): `fringe` 9 clumps unmirrored, from her right to her left, tips at
  the traced points (the middle one to the lash line between the eyes at 1.5636, the outer ones over the brows' ends at 1.579-1.589,
  a gap of forehead between the outer clumps and the central mass on each side), roots at the part a little to her right; `wisps`
  3 thin strands between them, a little longer; `temple` 2 a side from the part over the temple, above the ear, falling behind
  it; `crown` 4 a side over the top and back, down behind the ears onto the shoulders (the outer silhouette); `back` 5 a side
  to a V at mid-back (tips 1.246-1.31 head space = 1.17-1.24 m); `side` 2 a side in front of the ears, down the cheek, over the
  jaw's corner, onto the bust's outer edge; `under` 3 a side, darker, between the neck and the outer hair; `loose` 2 a side, thin
  strands crossing the cheek's edge to the collar. The cap (B2a's) thinner on the crown (16 -> 12 mm), now behind everything.
- **Traced widths, front close-up framing** (head mm, the outer silhouette; the drawing / B2b / B2c final): 1.6026 (40 mm over the
  eye line) 202 / 174 / 190; 1.5826 210 / 163 / 188; **eye line 222 / 151 (68 %) / 208 (94 %)**; 1.5406 231 / 150 / 228;
  1.5206 239 / 149 / 235; **chin 246 (clipped by the frame) / 152 (62 %) / 234 (95 %)**; 1.4526 >= 251 / 156 / 237. The first build
  reached 247 at the chin (100 %); moving the locks behind the ears (below) cost 5 %, the envelope was widened to win most of it back.
  The temples (1.58-1.60) stay 7-10 % narrow: the drawing's hair stands 26 mm off the skull there.
- **Shading in the toon look** (`charkit/toonlook.py`, additive; `LOOK_TOON` in `beatrice.py`): the highlight band is now PER
  LOCK on a hair that carries `ring_off`/`hw` (`ring["per_lock"]`): each lock's piece is shifted by its own offset, bowed across
  it, serrated along the strands (`saw`, `teeth`) and narrowing to nothing at the lock's edges (a lens), so the "angel ring" reads
  broken lock by lock with sawtooth edges; B2a's single wave stays for a hair without the attributes. The lit hair is lighter and
  warmer (the ramp's top 1.9 -> 3.2 x, an ash brown), the under layer and the cap darker (`HAIR_DEEP`, by `hshade`), the
  streaks along the locks calmer (`HAIR_STREAK` 0.6 -> 0.3: B2a's read as busy stripes). Outlines: the hull now inks every lock
  standing off the one under it.
- **The fringe's shadow** (review item 7): crisp, one step, the clumps' points printed on the forehead and the upper lids --
  the key's shadow map at 0.2 mm texels and filter 2.0 -> 0.6 (`lights` option `res`, new in `toonlook.lights`). It applies to
  every cast shadow of the key (the chin on the neck, the arms on the body): all crisper, as the toon look wants. The under-chin
  shadow was not deepened (not cheap without a new mask; left).
- **The crown's spike** (B2a/B2b's "antenna" standing on the crown in every turn): the egg's radius is 0^(-1/n) = infinite on its
  vertical axis (`Egg.rho`), and a column of the cap's grid landed exactly there; the cap's field nudges points off the axis
  (`pole_guard`, B2c's table only; B2a's hair kept as built).
- **The "ghost ear"** (review item 9): GEOMETRY, not a picture artefact, and not an ear: the eye opening's cut is a prism along the
  view axis through the whole head, limited only to "near the egg's surface", so it also cut an eye-shaped dent into the BACK of the
  skull at the eyes' (x, z) (the skull's radius at 1.575 m stepped 5 mm between 130 and 170 deg round). Found by ray casts (only the
  skin there) and renders with the ear, eyes, lashes, outlines, shadow proxy and the key's shadow switched off one by one (it stayed
  in all). The cut is now kept to the front (`EYES["front_only"]` = y < -0.03 head space, `Head._cut`): the skull is smooth there
  (radius 0.0825-0.0842 from 130 to 160 deg). The **ragged terracotta band** in the hairless profile is NOT the hidden hair's
  shadow: it stays with the key's shadow off -- it is the skin ramp's warm terminator on the bald skull (the face is shaded as a
  simple head only over the face); the hairless pictures are for the face, the skull is never seen.
- **Three worst defects after the first build, two attempts each** (harness renders `C:/tmp/beatrice_b2c/o/a1..a10`):
  1. *The hanging hair read as straight parallel cords and a flat curtain* (front: dangling noodles onto the chest; back: a wall).
     Attempt 1: a soft S per lock (`wave`), tips flaring out (`add`), ragged lengths (`z_jit`), the loose strands shortened to
     the collar. Attempt 2: the side locks wider below the jaw and moved out over the bust's edge, the back's tips pointed earlier
     and more waved. Better: a mass with separated waved ends, the front locks lie over the bust's edge; still straighter and more
     regular than the drawing's flyaway strands.
  2. *The fringe read as a comb of separate spikes* (a1: narrow teeth with skin between). Attempt 1: clumps 1.5x wider, kept wide
     down to 70 % of their length then a short point; the central five overlap as one mass with gaps only where drawn. Attempt 2:
     thin wisps between the clumps, calmer streaks. Now a mass with points and the two drawn gaps; the tips are still a little
     regular (similar triangles) next to the drawing's longer, finer points.
  3. *The crown read as a helmet with a blocky highlight* (B2a's smooth wave, then per-lock blocks). Attempt 1: the band per lock
     as a lens narrowing to the edges. Attempt 2: a near-continuous ring with serrated edges along the strands (smaller per-lock
     offsets, more teeth), the step along the locks 3 -> 1.8 mm. The ring reads as drawn anime hair; the dome's outline is still
     rounder and the part less visible than the drawing's.
- **After the first standard set: the elf ear** (`face_turn.jpg` 30-135 deg, `head.jpg`): a pale pointed tab above each ear -- a
  temple lock crossed the ear's top-back and cut its round top into a point, with bare skin between B2a's hairline (13-17 mm over
  the ear) and the locks. The temple, crown and under locks now leave the head 8-10 deg further back and fall behind the ear, the
  cap's hairline comes down to the ear's top (1.574), and the ear shows whole in profile as drawn, the hair starting right
  behind it. Two attempts (the hairline alone at 1.566 hid the ear's top half in profile; then 1.574 with the locks moved).

### Pictures, checks, reproduce

- `hair_vs_art.jpg` (new: the close-ups' framing -- drawing, the B2c hair, its outline over the drawing, B2a/B2b's hair and its
  outline, front and profile), `hair_back.jpg` (new: `with_hair_back.png` beside the hair straight from behind, three-quarter
  behind from both sides, and B2a/B2b's back), `face_vs_art.jpg`, `face_turn.jpg`, `face_closeup.jpg`, `head_size.jpg`,
  `look_compare.jpg`, `posed_vs_concept.jpg`, `posed_full.jpg`, `posed_hotlight.jpg`, and the full standard set (sheet, turnaround,
  every beauty and dev view) regenerated with the B2c hair. `face_turn` renders a 225 deg view too (`FACE_SHOTS`; the sheet keeps
  its six). `face_zebra.jpg` is B2b's (the face's surface did not change in front).
- **Nothing else moved** (`C:/tmp/beatrice_b2c/tools/cmp.py`, B2b's with the head skin compared by surface distance;
  `Beatrice_b2b_3292a4c.blend` against this build): 38 objects identical (max |dP| < 1e-7 m) -- every suit part including the
  collar, the hands and nails, the boots, and the head's ears, eyes and lashes. Rebuilt: `Beatrice_Hair_Cap` (162,800 -> 166,946
  vertices), `Beatrice_Hair_Locks` (58,456 -> 151,380). Changed: `Beatrice_Head` (the skin) ONLY behind the ears, y 0.050-0.090 m,
  z 1.567-1.590, up to 5.8 mm -- the eye-shaped dent at the back of the skull filled (the ghost-ear fix); everything in front of
  y 0.050 identical to 10 um.
- **The hair against the suit**: every lock's path is pushed 5 mm off the suit's outer surfaces (`body_clear`), then every lock
  vertex below the head lifted 1.2 mm clear along the suit's normal (`skin_clear`; about 2,000 vertices a build). Posed (the hair
  rigid with the head bone, `POSE["attach"]` unchanged): no lock shows through the body or the collar in `posed_full.jpg`,
  `posed_vs_concept.jpg` or the harness's posed front34 / back34 / side; the locks on the bust's edge lie on it.
- **Sheet scores** fall again with the wide hair against the BALD A-pose drawings (expected): above the waist front 0.837 -> 0.753,
  side 0.747 -> 0.680, back 0.817 -> 0.740; the bands below unchanged (side hip-knee 0.920 -> 0.922: the back's longest tips).
- `source/mesh_stats.json`: 62 objects, all closed, the flipped-normal list unchanged; the head 1.80 M evaluated triangles
  (B2b 1.61 M).
- Times: the build 8.4 min (the hair 20 s of it), the full set 12 min; the face views 1.5 min each (toon, pbr), the `hair_b2a` and
  `apose` variants 2-3 min each (their heads rebuilt), posing 1-1.5 min each; the hair loop
  `C:/tmp/beatrice_b2c/tools/hh.py` (the hair rebuilt on the B2b build, the look applied, the close-up framings and turns
  rendered) 50-70 s, `mont.py` the drawing / render / outline sheet and the traced widths.
- Reproduce (from `art/characters`): as B2b, plus `face_views.py -- --head hair_b2a --only art,turn` (the before columns) before
  `python Beatrice/scripts/face_sheets.py`.

### What still looks wrong in the hair (worst first)

1. **The crown reads as a rounded helmet** (`face_turn.jpg` 0 deg, `hair_vs_art.jpg` front, `posed_vs_concept.jpg`): the top's
   outline is a smooth dome with an inked rim, the temples flare round like a bonnet; the drawing's crown is narrower at the top,
   the part visible as a pale whorl with locks sweeping from it, and the sides fall in long flat curves. No part line reads.
2. **The hanging hair is too regular and too straight** (`hair_back.jpg`, `face_turn.jpg` 135/180, `posed_full.jpg`): a dense
   curtain of parallel clumps with even waves; the drawing's back is wider at the shoulder blades, layered, with separated
   flyaway strands and a clearer V. Its ends still stand a few cm off the upper back in profile (`turnaround.png` side).
3. **The fringe's tips are similar triangles** (`face_closeup.jpg` eyes, `hair_vs_art.jpg`): the drawing's clumps end in longer,
   finer, uneven points and fine single strands; ours are a row of short broad points. The fringe also stands well off the face
   in profile as drawn, which in three-quarter reads a little visor-like.
4. **Colour and lines**: the lit hair is a warmer, lighter brown now, but the drawing's lit side goes to a pale ash-beige and its
   strands carry fine dark lines; ours is one dark brown with a soft streak and outlines only on the clumps' edges.
5. **The side locks onto the chest** read as two dark cords in the concept framing (`posed_vs_concept.jpg`); the drawing's front
   strands are thinner and fade.
6. **The ear** still reads slightly pointed at 30-60 deg (`face_turn.jpg`): the ear's own field (tilted back, its top a narrow
   plate), not the hair now.
7. The face round's items (the eyes' lashes and lid shadow, the brows, the latex blobs on the upper chest) are untouched; the
   under-chin shadow was not deepened.

## B2d: the face finish and the latex sheen (2026-10-20, answers `previews/review_stage11.md` items 2-6 and 10)

The before is `source/Beatrice_b2c_e9e4b19.blend` (gitignored). Only the `Head` collection's face parts (skin, eyes, lashes,
ears) and the look change; the body and the hair stay (the hair is now laid on the frozen B2c face table, `FACE_B2C` /
`HEAD["hair_face"]`, so the face's field can move under it without moving it).

### Step 1, the eyes, brows, nose, mouth and blush: a drawn layer

Tools (scratch, `C:/tmp/beatrice_b2d/tools/`): `hd.py` (the face parts rebuilt on the B2c build -- eyes and lashes always, the
skin and ears on request, else their masks and materials re-made on the saved meshes -- the look applied, renders: art frames,
turns, eye close-ups at 0/30/60/90 deg, beauty views, `--zebra`; 55-65 s, 3.5 min with the skin), `art_over.py` (the canvas's
edges laid over the front close-up in its frame: the trace check), `grid.py` (the close-up on a 2 mm grid), `cmp_eyes.py`,
`curv.py` (principal-curvature maps of the skin field), `curvstat.py`, `cmp.py` (the vertex check), `sheets_b2d.py`.

- **Mechanism** (`charkit/animehead.py`, additive): `face_art` draws what the close-ups ink and paint on ONE front-projected canvas
  (head space, |x| 0..66 mm, z 1.495..1.592, 0.04 mm a pixel, straight-alpha RGBA; `Canvas.stroke` / `.field`, every edge
  anti-aliased by its distance), every shape from the new `FACE["art"]` table, every colour from the palette; the face material
  samples it over two new point attributes `art_u` / `art_v` (and `art_w`, the front-facing weight) and the image is packed into
  the .blend (`Canvas.image`). It REPLACES B2a/B2b's per-vertex brow, crease, lower lash line, nostrils, mouth line and nose
  highlight (those stay for a face without `art`): a vertex mask at 0.84 mm cannot draw a lash 0.3 mm wide. The upper lash band's
  centre line is now one function (`lash_path`) shared by the geometry and the drawn mass. The eye material reads a new `paint`
  table (`EYES["paint"]`) and two new eye-surface attributes, `ec` (the opening's distance, mm) and `ex` (along the eye).
- **Traced** (`art_over.py`, `grid.py`; head mm, x from the midline, z over the eye line): solid lash +6.6..+9.7 at 33.6 mm out,
  its strokes to +11.0, the crease +12.2 (4.9 over the lid); the outer mass reaches 57.6 mm at +5.2 (the geometry flick's end,
  kept); lower lash line about 1 mm under the lid; brows centre +18.0 at 16.5 mm out rising to +20.6 at 40 (B2b painted them
  2.5 mm higher), 2.3-2.5 mm thick; nose highlight -19.5..-32, 12-13 mm wide, glint at -30; nostrils -35..-38.5, 2.9-5.2 mm out;
  the mouth line -55.5..-56.5 with the lower lip tinted right under it, two gloss spots 3 mm either side.
- **Iris**: the brief asks 10 % larger; on the front close-up the width measures EQUAL (18.3 mm at the eye line on both the drawing
  and the B2c render, luminance runs), so review 11's 33 vs 37 px is not reproduced; the concept's three-quarter face does draw it
  fuller, so it was enlarged as asked (10.7 x 11.0 mm half-axes, B2b 9.7 x 10.0). The lid's shadow: on the drawing the visible
  iris's top half is 50-90 luminance, its lower half 115-160; B2c's top was 121-127; now 76-78 (`lid` over 12 mm, LID_SHADOW 0.85).
- **Corners**: the eye surface carried 3 mm under the lids (margin 1.8 -> 3.0 mm, its grid 4.5 mm past the opening); the inner end
  of the lash band thinner (it hooked down to the lower lid); the outer corner's walls inked (`art["outer"]`), the lower rim stops
  short of the corner (`rim_x`); the sclera greyer where it tucks under the lids (`ec`), the inner corner pink (`ex`).
- **Blush**: measured on the front frame (7x7 means): the drawing's cheek (227, 162, 149), B2c (236, 176, 161), B2d (234, 164, 150)
  -- the green drop now as drawn; wider and lower (`blush` sigma 19 x 10.5 mm) and across the nose's bridge (`blush_bridge`).
- **Attempts**: the eyes three (a1: the layer; a2: the inner wedge, greyer sclera, the brows lowered to the trace; a3: the iris
  shadow darker, nostrils, mouth), the nose highlight two plus one dropped (a per-vertex mask by the normal's sideways component,
  to keep the highlight off the nose's flanks in three-quarter, broke the front's highlight into streaks and erased the nostrils:
  removed), the mouth two.

### Step 2, the cheek groove (review 11 item 5)

`curv.py` mapped the skin field's smallest principal curvature over azimuth 0-100 x elevation -70..+25 deg. Cause: the front view's
side radius (`egg["side"]`, the traced cheek-to-chin outline) is nearly straight, and its spline's small wiggles were CONCAVE along
the elevation (-3..-5 /m at -21 and -38 deg), all the way round the side of the face -- the groove from below the eye to the jaw's
corner; and the socket's -2 mm dent and the cheek's +2 mm form had rings that made closed islands under the eyes. Fix (attempt 1,
kept): `egg["side_smooth"]` = 6 deg (the side radius smoothed along the elevation, `Egg._smoothed`, shared with `front_smooth`):
nowhere concave (min +1.2 /m), the outline moved 0.44 mm at most; the socket -1.5 mm and the cheek +1.3 mm, both wider. In the
cheek box (az 25-80, el -45..-5) the 5th-percentile k_min went -6.8 -> -2.5 /m, the side bands are gone; what is left concave is
next to the nose (the subnasal region, drawn so). `face_zebra.jpg` (B2c above, B2d below): at 30 deg B2c's reflection lines kink
into a V along the groove, B2d's run through; at 60 deg the far-cheek band is gone. **The hair is laid on the B2c face**
(`FACE_B2C`, `HEAD["hair_face"]`, new `animehead.build(..., hair_face=)`), so it did not move.

### Step 3, the ear

The ear's upper plate 2 mm lower and 1.5 mm wider backwards (`FACE["ear"]["plates"][0]`): a rounder top-back. At 30-60 deg it reads
less like a point, but the hair's hairline still crosses its top and makes a corner there (`face_turn.jpg` 60 deg) -- the hair's,
not changed.

### Step 4, the latex sheen and the knee pads (`charkit/toonlook.py`, additive; `LOOK_TOON` / `_LATEX` in `beatrice.py`)

- **The base** lifted and warmed (ramp 1.6 / 2.0 / 2.8 / 3.4, red over blue; B2b 1.0 / 1.2 / 2.2 / 2.8, blue).
- **A broad sheen** (`sheen`, `toonlook.sheen_lobe`): the view reflected about the normal against a fixed direction in CAMERA space
  (up and to the viewer's left, like the key) -- a matcap lobe: no light is needed, so it keeps its size at any distance (a lit
  glossy is a speck on a sun or a flood on a strip), and it turns with the view as the look's lights turn with the rig; gated by the
  key's light (no sheen in shadow) and fuller on rounder forms. Colour `SUIT_SHEEN` (a warm mid grey).
- **The crisp white kept to round surfaces**: `toonlook.curvature_attr` writes `toon_curv` (1/m, per vertex the largest normal
  curvature over its edges, smoothed) on the latex objects at prepare time (measured: upper chest 2-6, bust 9-25, flank 9-13, thigh
  9-21, sleeve 29-60); the streak layer is multiplied by its ramp 9..16. The upper chest's ragged blobs are gone (`head.jpg`,
  `face_turn.jpg` 0-60 deg), the flank's long white stripe is greyer and thinner (not broken).
- **Knee pads**: their own sheen facing the camera (`LOOK_TOON["materials"]["pad"]`): a soft graded highlight on the dome under the
  bright rim (`knee.jpg`).
- **Measured** (luminance percentiles 25/50/75/90/95/99, the bust box of the posed concept framing, background masked): the
  concept 32/45/113/201/213/251; B2c 27/32/58/61/75/232; B2d 30/45/56/89/117/216. The mid tones exist now and the median is the
  concept's, but the concept's sheen is far brighter and broader (90th 201 against 89). A second attempt (lobe threshold 0.74 ->
  0.66, the sheen colour lighter) gave 30/48/63/95/122/220: little more; reverted to keep the rendered set consistent. To reach the
  art the sheen needs to be a shape painted per form (a soft band along each breast's upper half, a gradient over the shoulders),
  not one lobe for the whole suit.

### Pictures, checks, reproduce

- New: `eyes_b2d.jpg` (the art's eyes -- front close-up, concept.jfif's three-quarter for 30 and 60, the side close-up -- B2c, B2d at
  front / 30 / 60 / profile), `latex_b2d.jpg` (B2c | B2d: portrait framing, the posed concept framing, full figure, knee; the B2c
  hero was rendered without the floor), `face_zebra.jpg` (now B2c | B2d). Regenerated: `face_vs_art.jpg`, `face_turn.jpg`,
  `face_closeup.jpg`, `head_size.jpg`, `look_compare.jpg`, `hair_vs_art.jpg`, `hair_back.jpg` (the face changed in them, the hair
  not), `posed_vs_concept.jpg`, `posed_full.jpg`, `posed_hotlight.jpg`, the full standard set.
- **Nothing else moved** (`C:/tmp/beatrice_b2d/tools/cmp.py`, B2c's with the hair compared by index; `Beatrice_b2c_e9e4b19.blend`
  against this build): 37 objects identical (max |dP| < 1e-7 m) -- every suit part, the hands and nails, the boots, AND
  `Beatrice_Hair_Cap` / `_Hair_Locks`. Changed, the Head collection's face parts only: `Beatrice_Head` (the skin, up to 1.41 mm: the
  socket, cheek and side smoothing), `_Ear` (up to 3.3 mm, its top), `_Eye` (carried further under the lids; 20,680 -> 25,076
  vertices), `_Lash` (its inner end thinner, up to 0.65 mm).
- **Sheet scores** unchanged from B2c (above the waist front 0.753, side 0.680, back 0.740; below unchanged).
- `source/mesh_stats.json`: 62 objects, all closed.
- **Times**: the build 494.5 s (8.2 min; the canvas about 8 s of it), the full set 692.8 s (11.5 min); face views toon 84 s, pbr
  89 s, `apose` 166 s; posed toon 61 s, pbr 58 s, hot 18 s; `hd.py` 55-65 s (3.5 min with the skin).
- Reproduce: as B2c (`run.sh ... --sheet --turn --save --beauty <the list>`, `face_views.py` toon / `--look pbr` / `--head apose
  --only turn`, `posed.py` toon / `--look pbr` / `--look pbr --views concept --lights hot`, `posed_sheets.py`, `face_sheets.py`),
  then `C:/tmp/beatrice_b2d/tools/hd.py -- --blend <Beatrice.blend> --tag b2d --keepall --views e0,e30,e60,e90` and
  `-- --tag z_b2d --keepall --zebra --nohair --views t0,t30,t60` (and the same on the B2c .blend for the before columns), then
  `sheets_b2d.py`.

### What still looks wrong (worst first)

1. **The nose in three-quarter** (`face_turn.jpg` 30-60 deg, `face_closeup.jpg` three-quarter, `head.jpg`): the front-projected
   highlight lands on the nose's flank as a large white patch, and one nostril stroke shows as a comma on the side. The front view is
   right. It needs the highlight to be view-aware (a separate layer faded by the view's angle to the face) -- not done.
2. **The latex sheen is weaker than painted** (`posed_vs_concept.jpg`, `latex_b2d.jpg`; numbers above): mid tones now, but the
   concept's bust is half grey sheen; the flank streak is still one long line.
3. **The blush reads strong at portrait distance** (`face_turn.jpg` 0 deg, `head.jpg`): it matches the drawing's colour measured at
   the cheek, but the drawing's skin is darker overall (229, 186, 171 against our 241, 203, 185), so ours reads redder against it.
4. **The eyes**: the lash mass is right in shape and weight from the front; its strokes are finer and more comb-like than the drawn
   soft brown feathering, and the outer flick is still a solid wedge (`eyes_b2d.jpg`). The iris lacks the drawing's layered
   texture. In profile the eye is the drawn convention but small.
5. **The mouth in profile**: no pout was modelled (paint only); the profile runs nose to chin as before (`face_closeup.jpg`).
6. **The ear's top** still meets the hairline in a corner at 60 deg (the hair's).
7. The skin's light tone and the fringe's soft shadow streaks on the forehead (review 11 item 7, the hair round's) are unchanged.

## B2e: the hair's colour, crown and silhouette, the nose, the head's pose (2026-10-22, answers `previews/review_stage12.md` items 1, 2, 3, 6, 7)

The before is `source/Beatrice_b2d_09b46cb.blend` (gitignored). Only the `Head` collection's hair objects change in the
build; the face's paint (the nose layer), the look and the posing change at render time.

### Step 1, the drawn colours (read first)

Sampled on `concept-face-front-v1.png` (boxes in its 1254 px frame, luminance percentiles of the hair pixels; tool
`C:/tmp/beatrice_b2e/tools/mont.py`, `colour_sheet.py`) and `concept.jfif`:

| | drawing (front close-up) | concept.jfif | B2d model |
|---|---|---|---|
| lit side, her right, outer layer (p50 / p75) | (184, 154, 145) / (203, 178, 169) | (166, 143, 138) / (207, 194, 190) | (43, 31, 29) / (68, 50, 47) |
| pale edge (p95 of the lit side) | (212-216, 190-215, 180-220) | up to (254, 254, 254) | (74, 55, 52) |
| mid tone, her right low (p50) | (138, 109, 101) | -- | (42, 31, 29) |
| shadow: the fringe (p50) | (66, 51, 50) | -- | (72, 54, 50) |
| shadow: her left side (p50) | (56, 45, 45) | (52, 46, 50) | (48, 35, 33) |
| strand lines (p5 of the lit side) | (64, 47, 45); darkest (20, 12, 12) | (46, 30, 29) | (25, 17, 16) |
| angel ring / gloss on the dark side (p95, crown her left) | (77, 71, 79), a cool violet grey | -- | (144, 127, 122), flat blocks |
| lit crown, her right of the part (p50 / p95) | (199, 172, 164) / (234, 216, 209) | (204, 183, 176) p75 | (73, 54, 51) / (144, 127, 123) |

What the drawings do (and the model now does): the hair is the dark brown everywhere EXCEPT the outer layer of her right side
seen from the front, which goes warm ash to a pale ash-beige towards its edge, with dark strand lines along it; the fringe and
the face-framing locks stay dark even on her lit side; the side (`concept-face-side-v1.png`, `with_hair_side.png`) and back
(`with_hair_back.png`) drawings are dark brown all over with soft glossy bands. A real light cannot do both (a light from her
right lights the profile too), so the light on the hair is PAINTED.

The part: the drawn whorl is at column 635 of the front close-up, the face's midline at 648-653 (the irises' midpoint 648, the
model's midline 653): 3 mm to her right -- B2c's reading (-4 mm) confirmed; the reviewer's "just right of centre" is the
viewer's right of the image's centre, the same place.

### Step 2, the hair's colour: the light as painted (`charkit/toonlook.py`, `charkit/animehead.py`, `charkit/animehair.py`, additive)

- **The tone** (`LOOK_TOON["materials"]["hair"]`): the ramp now runs from the drawn shadow (52, 40, 40) through the mid tone
  (95, 73, 69) and the warm ash (165, 135, 126) to the pale edge, as multipliers of the dark-brown base (B2c: 0.95 .. 3.2, so
  every lit part stayed a dark brown). What drives it is no longer the key light alone: **`toonlook.normal_lobe`** -- the
  shading normal in CAMERA space against a fixed direction (the viewer's upper left, as the art's light), mixed into L (weight
  0.85), gated by (a) `side`: only on her right of the head (`toon_x`, written by `ring_attr` before posing), (b) `world`: only
  where the surface faces her right / up in the world (seen from her right, the profile, the camera-facing side stays dark,
  as the profile close-up paints it), (c) `view`: only while the camera looks from the front or her right (from her left,
  the lit side is the far one and read as a pale bun behind the crown), (d) `per_lock`: each lock's `lit` share along it (new
  per-vertex attribute `hlit`, row key `lit`: the outer layer -- temple, crown, back, the under layer -- takes it; the fringe only
  at its roots on her right of the part; the face-framing side locks and the cap not). The hanging hair is shaded as one mass for
  this (`toonlook.radial_normals` `hang`: below the jaw the normals blend to the horizontal from the head's axis), so a curtain's
  side reads as the mass's side, not ribbon by ribbon. A real light was tried first (a hair-only key via light linking, which
  EEVEE 4.5 supports): it lit the profile too and the inner faces of her left curtain, neither of which the drawings do.
- **Strand lines** (`animehead.materials`, palette `HAIR_LINE`, `_SCALE`, `_W`, `_AMOUNT`): thin dark lines along each lock,
  the isolines of a noise stretched along it (380 per metre across, 5 along: 2-3 mm apart, long).
- **The angel ring**: now ADDED to the tone (`ring["add"]`, `HAIR_RING` a cool grey #6e6670, B2c mixed to a flat grey-beige
  #9c8b86), broken along the strands (`ring["strands"]`, the same stretched noise) and a little of it kept on the shadow side
  (`lit_floor` 0.45: the drawing's cool violet sheen on her dark left).
- **The pale edge**: a facing rim on the lit side, `HAIR_EDGE` #d8c8c0.

### Step 3, the crown, the part and the silhouette (`HAIR_B2E`; `animehair` `proot`, `part`)

- **The part** (`HAIR_B2E["part"]` = -5 deg, her right): the fringe, temple, crown and back locks are rooted ON it (`proot`:
  beta, azimuth from the part; mirrored rows take the opposite offset), so they spring away from it on both sides; her right of
  it lit (the drawn pale side of the part), the fringe and her left dark. The fringe's nine clumps root along the part's two
  sides over the front of the crown (rooted all at the whorl in the first build, they fanned out like a broom).
- **The temples** (review: 88 % of the drawing at brow level, a rounded rim): the temple and crown locks stand off the skull
  (offset to 10 mm at the leave point plus an 8-10 mm bulge; B2c 5-7 mm and 0.007 at the end) and leave the head higher (beta
  20-30, B2c 2-16), so the mass widens steadily from the crown down; the envelope re-traced at 1.58-1.62. Traced widths, front
  close-up framing (head mm, drawing / B2c / B2e): 1.6026 202 / 190 / 198; 1.5826 210 / 188 / 213; **eye line 222 / 208 / 230**;
  1.5406 231 / 228 / 243; 1.5206 239 / 235 / 249; **chin 246 / 234 / 252**; 1.4526 >= 251 / 237 / 254 -- at the brows 101 %
  (B2c 90 %), at the eye line 104 % (94 %), at the chin 102 % (95 %).
- **The back** (review: a waist where the crown's shell met the hanging hair, a V notch): the back locks root at the crown's
  back from the part (no gap at 180 deg) and leave the skull at beta 8 .. -8 (B2c -6 .. -22, at the nape), the envelope's back
  depth kept from 1.60 down (0.114-0.117); `hair_back.jpg`.
- **The fringe**: its outer clumps now to the lash line (tips 1.5698-1.5742, B2c 1.5790-1.5890: the brows), every clump its own
  point length and end width (`_taper(at=, end=)`), the central five 25 % wider (one mass with points), four wisps between.
- **The chest locks** (review: cords): flat and wide -- 2.4 mm thick (B2c 4.5), up to 30-32 mm wide (21-24), curl 0.10 (0.30),
  a longer taper.
- **The hanging ends**: length jitter 20-30 mm (B2c 8-12), the back's tips at different heights and taper points.
- **The "wires" at the chin**: B2c's two thin `loose` strands a side are left out (at 30-60 deg they stood off the chin).
- **The ear** (review item 7): the cap's hairline arcs over the ear 3-5 mm above its round top (B2c's line at 1.574 cut the top
  flat and its thick edge crossed it on a diagonal at 60-90 deg), the edge tapered over 10 mm (6: a brim); the side locks root
  lower, under the temple lock. In profile the ear now shows its round top; a thin strip of skin shows between it and the hair.

### Step 4, the nose (review item 3; `animehead.nose_art`, `FACE["art"]["nose"]["view"]`)

The drawn highlight, measured on the front close-up: a soft round glow with a slightly pointed top, 12.9 mm wide, z
1.5305-1.5423, centred about 4-5 mm to her RIGHT of the nostrils (the lit side of the tip), near-white at the core (253, 247,
243) going warm at the edge (250, 230, 220), a small glint low on its left; a faint warm shade round its foot. B2d's: a flat
hard-edged trapezoid on the midline. Now the highlight is its own layer (a second front-projected canvas with SIGNED x -- the
skin gets a new attribute `art_x`, its vertices unchanged), soft-edged (`hi`: centre (-3.5 mm, 1.5368), radii 4.7 x 5.4 mm,
35 % of the radius soft), with a 0.6 x 0.9 mm glint and a warm `halo`; the material thresholds its intensity by how square the
view is to the face (the shading normal against the view, `fade` = 0.55 .. 0.95, 0.60): the whole glow from the front, its
core at three-quarter from her right, nothing on the flank at 30-60 deg from her left or by 60 deg from her right. The
nostrils a touch lighter (`NOSTRIL` #9a5a50). Two attempts at the fade (0.82 .. 0.98, then 0.55 .. 0.95; from her left at 30 deg the glow lies on the far side of the
tip either way, so it is simply not seen there). `face_closeup.jpg`, `face_turn.jpg`, `face_vs_art.jpg`.

### Step 5, the head's pose in the concept framing (review item 6; `POSE`, `charkit/suitpose.py`)

Measured on `concept.jfif`: the irises at (427, 340) and (570, 317) -- her right eye 23 px lower over 143 px, a 9 deg tilt to
her right shoulder; the brows low over the eyes and the face short from eyes to mouth (0.72 of the eyes' spacing; the model's
straight-on 0.82): the chin down. The B2d render's eyes: (474, 323) and (597, 326) -- 1.5 deg the other way. Now `head` =
(10, -9.5, -2) deg (chin down 10, her right side down 9.5; B2d (0, 1, -2)): the render's eyes (460, 363) and (581, 345), 8.7
deg. **The gaze** is a pose parameter (`POSE["poses"][...]["gaze"]` = (out, up) in iris radii, `suitpose.gaze` moves the eye
surfaces' `iu`/`iv` after posing; the lids and their shadow stay): (0, 0.10), the iris raised under the upper lid, so she
looks up at the camera from under the brows. `concept_b2d` is the same pose with B2d's head, for the before/after.
**The hair with the head** -- done differently from the brief: rigid with the head bone to its tips (B2a-B2d), the 10 deg nod
swung the chest locks about 4 cm back into the suit (the locks hang 25 cm below the head's pivot) and a lock along the jaw
stood in front of the chin (the chin lies below the head's weight band and does not nod). A head-to-chest blend by height was
tried and still crossed the chin. Now the hair moves by the trunk's height bands, exactly as the skin, collar and suit at its
height (`attach={"Beatrice_Hair*": "trunk"}`): rigid with the head above the jaw (above 1.53 m), with the neck and chest
below. In `posed_full.jpg`, `posed_vs_concept.jpg` and the harness's posed front / front34 / side / back34 no lock shows
through the suit, the collar or the face.

### The three worst defects after the first full build (two attempts each)

1. *The lit side read as a separate pale patch* -- from 22 deg to her left (`head.jpg`) the far, lit side stood behind the dark
   crown like a pale bun. Attempt 1: the temple locks' roots and mid points lower; attempt 2: the painted light only while the
   camera looks from the front or her right (`view` axis (-0.35, -0.94, 0)): from her left that side is the far one. Fixed.
2. *The fringe fanned from one point like a broom* (all nine clumps rooted at the whorl), skin showing between narrow spikes.
   Attempt 1: rooted along the part's two sides over the front of the crown (beta 76-86, azimuth -32 .. +28 from the part);
   attempt 2: the central five 25 % wider, the wisps rooted with them. Now one mass with points and the two drawn gaps.
3. *Her right of the part dark* (the drawn pale side of the part missing: crown box p50 59 against the drawing's 199). Attempt
   1: the side gate narrowed to the head's midline (x 0 .. -20 mm; the crown's top lies within 10 mm of it); attempt 2: her
   right fringe clumps take the light further down from their roots (`_FRINGE_LIT_R`). p50 131, p75 184 (drawing 199, 215).
   Also the side locks' roots take the light of the mass round them (`lit` (0, 0.8) -> (0.3, 0.15)).

### Pictures, checks, reproduce

- New: `hair_colour_b2e.jpg` (the drawing's, B2e's and B2d's hair sampled in the same boxes of the front close-up framing --
  lit side, pale edge, mid tone, the fringe's and her left's shadow, strand lines, gloss -- then front crops: drawing, B2e, B2d,
  concept.jfif, B2e at 30 deg; the drawing's "pale edge" sample catches its backdrop, the B2d crop its sheet label).
  Regenerated: `hair_vs_art.jpg` (B2e against B2c's hair in the B2e look, `--head hair_b2c`), `hair_back.jpg`,
  `face_vs_art.jpg`, `face_turn.jpg`, `face_closeup.jpg`, `head_size.jpg`, `look_compare.jpg` (the nose and hair changed them),
  `posed_vs_concept.jpg` (now with B2d's head pose as a fourth panel, `--pose concept_b2d`), `posed_full.jpg`,
  `posed_hotlight.jpg` (its toon panel no longer labelled "B2b"), and the full standard set. `eyes_b2d.jpg`, `face_zebra.jpg`,
  `latex_b2d.jpg` are B2d's (unchanged parts).
- **Colours, the front close-up framing** (drawing / B2e / B2d, `hair_colour_b2e.jpg`): lit side (192, 163, 154) / (183, 157,
  149) / (66, 49, 46); pale edge -- / (208, 182, 174) / (74, 57, 54); mid tone (136, 107, 100) / (99, 77, 72) / (42, 31, 29);
  fringe shadow (66, 51, 50) / (56, 43, 43) / (72, 53, 49); her left (53, 43, 43) / (53, 41, 41) / (41, 29, 28); strand lines
  (62, 45, 43) / (37, 28, 28) / (27, 17, 15); gloss (78, 72, 80) / (78, 66, 70) / (144, 127, 123).
- **Nothing else moved** (`C:/tmp/beatrice_b2e/tools/cmp.py`, B2d's; `Beatrice_b2d_09b46cb.blend` against this build): 39
  objects identical (max |dP| < 1e-7 m) -- every suit part, the hands and nails, the boots, AND the head's skin, ears, eyes and
  lashes. Changed: `Beatrice_Hair_Cap` (166,946 -> 165,084 vertices: the hairline over the ear, the 10 mm fade) and
  `Beatrice_Hair_Locks` (151,380 -> 150,268; 45 locks, B2c 48).
- **The hair against the suit and skin** (`C:/tmp/beatrice_b2e/tools/gap.py`: lock vertices below 1.50 m within 20 mm of the
  suit's surfaces or the skin, every third, signed along the nearest normal): at rest 4,308 checked, 2 more than 0.5 mm inside
  (one 2.1 mm, at the nape under the collar, hidden); in the concept pose 4,394, one 0.51 mm inside, none deeper. 2,240 lock
  vertices lifted clear of the suit at build (B2c's pass).
- **Sheet scores** (the BALD A-pose drawings): above the waist front 0.753 -> 0.734, side 0.680 -> 0.651, back 0.740 -> 0.722
  (the wider temples and crown); the bands below unchanged.
- `source/mesh_stats.json`: 62 objects, all closed.
- **Times**: the build 484 s (8.1 min; the hair 18 s), the full set 685 s (11.4 min); face views toon 88 s, pbr 90 s,
  `hair_b2c` 236 s, `apose` 166 s; posed toon 62 s, pbr 59 s, hot 17 s, `concept_b2d` 34 s; the hair harness
  `C:/tmp/beatrice_b2e/tools/hh.py` (the hair rebuilt on the B2d build, `--face` the face's material, `--pose` posed) 60-110 s.
- Reproduce (from `art/characters`): `bash charkit/run.sh Beatrice --sheet --turn --save --beauty <B2b's list>`, then
  `C:/tmp/beatrice_b2e/tools/pics.sh` (face views toon / pbr / `--head hair_b2c --only art,turn` / `--head apose --only turn`,
  posed toon / pbr / pbr hot / `--pose concept_b2d --views concept`, `posed_sheets.py`, `face_sheets.py`), then
  `colour_sheet.py $TEMP/Beatrice_face <B2d's face_vs_art.jpg> previews/hair_colour_b2e.jpg`.

### What still looks wrong (worst first)

1. **The lit side is lock-by-lock**: where a lit lock meets a dark one the step is hard and follows the lock's edge (`face_turn.jpg`
   0 deg, `hair_vs_art.jpg` front); the drawing grades from ash to dark across the mass, and its lit locks carry darker strands
   inside them. The strand lines are finer and fainter than drawn on the lit side.
2. **In the concept framing the lit side is smaller than painted** (`posed_vs_concept.jpg`): the concept's whole outer mass on her
   right is pale beige to the chest; ours is ash at the top and mid-brown below.
3. **The crown's gloss on the dark side** is a row of small blocks across the fringe's top (`hair_vs_art.jpg` front, `face_turn.jpg`
   30 deg); the drawing paints one broad, curved, glossy band.
4. **The ear at 60-90 deg from her left** (`face_turn.jpg` 60, 90): its top is round now, but the hair's edge behind it still meets it
   in a corner, and in profile a thin strip of skin shows between the ear and the hairline.
5. **The hanging hair is still a fairly even curtain** (`hair_back.jpg`): ends at varied lengths now, but the drawing's separated,
   wavy, flyaway strands are not there.
6. **The pbr look's nose** (`look_compare.jpg`, `face_closeup.jpg` pbr): Cycles' own highlight still makes a pale strip down the
   bridge, as in B2d (the drawn layer is now the softer glow under it); the toon look, the default, is right.
7. The face round's items (the lash mass, the blush a fifth too strong, the skin a touch light, the under-chin) and the latex's
   sheen are unchanged.

## VRoid spike: Beatrice on a free anime base (2026-10-22, branch `33-beatrice-vroid`)

The owner's verdict on the merged B2e build: the toon look is "great", the shapes "too hard"; asked for a remake on a free
base (faces, hairstyles). During the spike (2026-10-22): the scratch body is "acceptable, but far from what I wished for --
I think the VRoid figure is better", so **variant B (the whole VRoid figure) is the main build** and A (the VRoid head on the
scratch body) is demoted; and **only four references count**: `concept.jfif`, `concept-face-front-v1.png`,
`concept-face-side-v1.png`, `concept-upper-body-side-v2.png` -- the A-pose sheets (`with_hair*`, `without_hair*`) are not
used (no fitting, measuring or comparing against them).

### Downloads and licences (checked 2026-10-22)

- **The page**: https://opengameart.org/content/vroid-studio-cc0-models -- licence field "CC0", author "VRoid Project",
  submitted by hecko 2024-12-24. Its text: "These models come from the alpha versions of VRoid Studio and were explicitly
  released by VRoid under CC0. They were downloaded from here where possible: https://vroid.pixiv.help/hc/en-us/articles/4402614652569
  ... (AvatarSamples A through C are missing because they were released under different, more restrictive terms.
  AvatarSamples D through G are unrelated to the samples with the same names in the current version.) AvatarSample_D_Darkness,
  the bases, and the hair samples were exported from VRoid Studio 0.14.0."
- **The licensor's page** (VRoid help, read 2026-10-22): "CC0 license models ... Base models; Sample models: HairSample_Male,
  HairSample_Female, beta Ver AvatarSample_1 .. _4" (each beta sample's own article: "The CC license for this model is CC0 ...
  pixiv Inc. has waived all copyright"). Beta AvatarSample_1 is Sendagaya Shibu (bob, P hairpin; a Darkness version); her
  sister Sendagaya Shino and cousin Sakurada Fumiriya are NOT on the CC0 list.
- **The files' own VRM meta** (`licenseName`) -- the deciding check, because it disagrees with the page for most files:

  | zip (sha256) | .vrm (sha256) | VRM title / exporter | meta licence | status |
  |---|---|---|---|---|
  | base_female.zip `bf2717cf...68eb3` | Base_Female.vrm `1995f174...8149b` | Base_Female / VRoid 0.14.0 | CC0 | CC0 by both |
  | hairsample_female.zip `e00fc661...bbe4` | HairSample_Female.vrm `d9ef39f0...ebda2` | HairSample_Female / VRoid 0.14.0 | CC0 | **CC0 by both -- USED, committed** |
  | avatarsample_d_darkness.zip `330ab409...7fa1` | AvatarSample_D_Darkness.vrm `037951ea...2a` | AvatarSample_D / VRoid 0.14.0 | CC0 | CC0 by both |
  | avatarsample_d_0.zip `311a383b...3167` | AvatarSample_D.vrm `9adf1b44...f9b7` | 千駄ヶ谷渋 (Shibu) / UniVRM 0.44 | Other (VRoid Hub terms) | CC0 by VRoid's page only |
  | avatarsample_e.zip `3b6808e2...b6b1` | AvatarSample_E.vrm `aea824cc...7911` | ビビ / UniVRM 0.44 | Other | not CC0-confirmed |
  | avatarsample_f.zip `e187569e...bd1` | AvatarSample_F.vrm `d8eff1ed...21bb` | ヴィータ / UniVRM 0.44 | Other | not CC0-confirmed |
  | avatarsample_g.zip `3dc643aa...1b34` | AvatarSample_G.vrm `ad5750ce...c0` | ヴィクトリア・ルービン / UniVRM 0.44 | Other | not CC0-confirmed |
  | sendagaya_shino.zip `ff85ff42...a3696` | Sendagaya Shino.vrm `f11b2648...d2a9` | 千駄ヶ谷篠 / UniVRM 0.44 | Other | not CC0 (not on VRoid's list) |
  | sakurada_fumiriya.zip `f3c53eea...fc66` | Sakurada Fumiriya.vrm `a36e91b8...83f3` | 桜田史利矢 / UniVRM 0.44 | Other | not CC0; a MALE model |

  Full hashes: `C:/tmp/beatrice_vroid/` (scratch) and `vroid/LICENSE.txt` for the committed file. The "Other" files carry the
  VRoid Hub licence URL (everyone may use, modify, redistribute, commercially, no credit) -- permissive, but not CC0, so by the
  brief's rule none of them enters the repo; they were downloaded, imported and rendered for the candidate picture only.
- **The importer**: "VRM format" add-on v4.10.2 by saturday06 (Isamu Mogi; maintainer field; copyright 2020 Isamu Mogi, 2018
  iCyP), extensions.blender.org, licence SPDX MIT + GPL-3.0-or-later, `add-on-vrm-v4.10.2.zip` sha256
  `228e622d158fbbf070b1b49547708f360a223873f00fa81a42170794492e6db9` (matches the extensions API's `archive_hash`), Blender
  4.2-5.3. Read before installing: Python, five .blend shader libraries and docs; no network, subprocess or ctypes use (urllib
  only for glTF URI parsing, base64 for glTF data URIs). Installed with Blender 4.5.10's own command line:
  `blender --command extension install-file -r user_default --no-prefs add-on-vrm-v4.10.2.zip` (the user's preferences left
  read-only; `vrmbase.enable()` turns it on per process under `--factory-startup`).

### Step 1, the candidates (`previews/vroid_candidates.jpg`)

Every female candidate imported (`vroid.py --candidates`, 3 min 52 s for eight, Sakurada Fumiriya left out: a male model),
its head rendered front and 30 deg as imported (Cycles, the studio rig) and in the toon look (the VRM's MToon materials rebuilt
as Principled on their lit texture, `vrmbase.materials`, then `toonlook.apply` with `LOOK_VROID`), beside the two close-ups and
concept.jfif.

- **The faces**: Base_Female, HairSample_Female and AvatarSample_D/D_Darkness share VRoid 0.14's default female face (the
  same 2,054-vertex mesh); the samples differ in the painted eyes. D_Darkness has elf ears and pink-white skin; E/F/G are
  stylised (big jewel eyes, odd colours); Shino's is the default face with amber eyes. None is as mature as the drawings
  (smaller, narrower eyes; a longer lower face); the default face with its eyes recoloured is the closest, and it comes with
  the CC0 files.
- **The hair**: Sendagaya Shino's (long, straight, dark, a blunt fringe, side locks to the chest) is by far the closest
  silhouette, but its file is not CC0. Of the CC0 files only HairSample_Female has a usable fringe -- pointed clumps to the
  lashes with a lock between the eyes, as drawn -- on a short back with cat ears and twin tails; Base_Female has only a skull
  cap, D_Darkness a pink bob.
- **Chosen**: face, body AND hair from ONE file, `HairSample_Female.vrm` (CC0 by both statements): the face for being the
  default face with the best fringe over it, the hair because its fringe is the drawn one and its back cards can be lengthened
  by script (step 2). Committed as `vroid/HairSample_Female.vrm` (17.9 MB, unchanged) with `vroid/LICENSE.txt`.

### Step 2, variant B: the whole VRoid figure (`scripts/vroid.py --build B`)

One command rebuilds it from the committed file: `"C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b
--factory-startup -P Beatrice/scripts/vroid.py -- --build B` (from `art/characters`; 26-34 s, saves
`source/Beatrice_vroid_B.blend`, gitignored). Mechanism `charkit/vrmbase.py` (new), tables `scripts/vroid_tables.py` (new),
driver `scripts/vroid.py` (new). `beatrice.py` is only read (its palette, `LOOK_TOON`'s lights and latex), never changed.

- **Import and tidy** (`vrmbase.load`, `split_face`, `drop`): the add-on's spring-bone colliders and MToon outline modifiers
  removed; the face's painted layers (eye whites, irises, highlights, lashes, eyelines, brows) split off into their own object
  (`Vroid_FaceLayers`) so the inked outline never rings a transparent card; the sample's clothes deleted (VRoid exports the
  body whole under them); the mouth's inside cards left out (the closed mouth showed a tooth through them).
- **The hair** (`HAIR_B`, `vroid.hair_b`, `vrmbase.drape` / `trunk_weights`): the 79 hair pieces (one card per lock, front
  and back faces) classified -- 38 removed (the cat ears and their inner cards: above 1.655 m or the ears' material; the twin
  tails: hanging below 1.385 m off the midline), 18 back and 27 face-framing pieces lengthened: each card's part below 1.50 m
  (back) / 1.47 m (sides) stretched down so its end reaches 1.12 +- 0.035 m (mid-back) / 1.21 +- 0.02 m (the chest), flared
  away from the head's axis and pushed out of the body (12 / 10 mm, the push carried down the lock so it bends instead of
  folding); the texture stretches along the strands, which hides it. Then the whole hair is re-weighted to the trunk by height
  (head above 1.43 m, neck, upper chest, chest), the B2e lesson: hair rigid with the head goes into the body on a nod.
- **The face** (`face_b`, `vrmbase.remap_z`): the lower face 10 % longer (every face vertex below the eye line moved down
  in proportion to its depth below it, every expression shape key moved the same way); the expression from VRoid's own keys
  (`EXPRESSION`, chosen from two rounds of trials, `FACE_TESTS_1` / `FACE_TESTS`): mouth neutral 0.7 + angry 0.3 (a level
  mouth -- the neutral VRoid mouth turns up at the corners), eyes angry 0.25 + close 0.16 (heavier lids), brows angry 0.3,
  the eye highlight half hidden. The painted eyes, brows and lashes are VRoid's own, recoloured.
- **The colours** (`RECOLOUR`, `vrmbase._recolour`): the face's painting tinted so its mean is the drawn skin (#f2d0c2); the
  body one flat skin colour (its texture paints underwear, which the leotard's high cut would show); the hair's luminance,
  normalised to its own 5-95 percentiles, mapped onto the drawn dark brown (#221a1b -> #45363a -> #76625e); the irises
  onto the drawn blue-grey (#2c3346 -> #5f6f8c -> #a4b2cc).
- **The suit** (`SUIT_B`, `suit_fields`, `vrmbase.shell` / `cut_material` / `collar`): what the four references draw -- a
  glossy black leotard with long sleeves, a stand collar and a zip to the lower belly, cut high over the hips (bare hip and
  outer thigh), glossy black thigh-high stockings, mesh inserts showing the skin (collarbone to armpit, and the side of the
  ribs down towards the waist), thin piping. NB this is NOT the merged build's full-length suit with knee pads and boots, which
  followed the A-pose sheet; with the four references only, the leotard is the drawn design. Built as two layers over the
  body (subdivided twice, 107,678 vertices): copies of the skin pushed out 1.6 mm (latex) and 1.0 mm (net), rigged as the
  body, cut by signed fields (metres, from the skeleton's landmarks, in the rest T-pose: the sleeve's end, the collar line,
  the leg openings' curves front and back, the stockings' top, the inserts as capsules in the front projection with a facing
  test, the zip strip) that the material thresholds crisply and draws the piping along; the stand collar is real geometry
  (`vrmbase.collar`: rings sampled off the neck's own cross-section, 6 mm off it, flaring 10 mm at the base, the front 12 mm
  lower, 1.8 mm thick). **Left out**: boots and knee pads (the references crop above the knee: the stockings run to the
  toes), heels.
- **Proportions** (`SCALES`): only the bust, 1.28 (the base's is small; both drawings draw it fuller). The VRoid body's
  slimness and leg length are kept (the owner's verdict). Height 1.63 m (the base's), not scaled.
- **The toon look**: `toonlook.apply` with its own table `LOOK_VROID` (`beatrice.LOOK_TOON`'s engine, world, lights and floor;
  its face, skin, latex and net specs; flat painted specs for the face's layers; the hair's ramp with B2e's painted light,
  `normal_lobe` with its side / world / view gates but not per lock, a cool gloss band, the hair's normals as one mass --
  one attempt: the effect is faint on the VRoid hair). `toonlook.prepare` runs on the opened build before posing (the
  hair's band and side attribute, the latex's curvature for the streaks). Two additive
  paths in `toonlook.py` (defaults reproduce today's output): `spec["alpha"]` (a cut-out / see-through layer: the Principled's
  Alpha between a Transparent BSDF and the emitted colour, dithered, transparent shadows) and `outline["masks"]` (a hull
  shown only where a point attribute is > 0, so a cut layer draws no line where it is cut away). Not ported: the face's
  simple-head normals and shadow proxy (the VRoid face's own normals are already smooth).
- **Posing** (`POSE_B`, `vrmbase.world_pose`, `vroid.pose_b`): the VRM skeleton posed by ABSOLUTE world rotations in
  `beatrice.POSE`'s convention, so the concept pose carries over (the arms brought down from the T-pose about Y first; the
  left thigh's inward swing reduced 8 -> 2 deg, as VRoid's legs stand closer); fingers relaxed; the standing foot put on the
  floor. **Found on the way**: a layer copied from the body must be PARENTED to the armature as the body is -- with the
  armature modifier alone, lowering the armature object to the floor left the suit 6 cm above the body.
- **The framings** (`FRAMES`, `vroid.framed_camera`): each drawing's camera solved on the posed model -- direction fixed, distance
  and aim iterated until two landmarks land on the drawing's pixels (concept.jfif: the eyes' midpoint and the crotch, residual
  9 px; the side drawing and the profile close-up: the height fitted on two marks, the aim on the eye / the nose's tip).

### Pictures and times

Timed end to end: the build 29 s, every B / what-if picture 75 s (five Blender runs), the sheets a few seconds.
`vroid.py --open B --pose concept --shots concept --tag P` (6 s) and `--open B --pose stand --shots
side_v2,art_front,art_side,face,p_front,p_34,p_side,p_back --tag S` (12 s), then `python Beatrice/scripts/vroid_sheets.py`:
`vroid_vs_concept.jpg`, `vroid_face.jpg`, `vroid_full.jpg`, `vroid_candidates.jpg`, `vroid_whatif_hair.jpg` (the what-if:
`--open B --whatif shino --pose concept --shots concept --tag W` and `--pose stand --shots face,p_back --tag W`, 10 s each,
reading Shino's file from the download folder). The B2e cells are cut from the merged build's own previews
(`posed_vs_concept.jpg`, `face_vs_art.jpg`, `face_turn.jpg`, `posed_full.jpg`), not re-rendered. Renders EEVEE 64 samples on
the RTX 3070: 0.5-1 s a shot after the first.

### What still looks wrong in B (worst first)

1. **The face is VRoid's child face, not the drawings' woman** (`vroid_face.jpg`): eyes too big and round under heavy lids,
   no blush, a pointed V chin, the nose a dot; the drawings' eyes are smaller, almond, with a soft pink blush, a fuller lower
   face. The lower-face stretch and the expression keys moved it a little; the face shape of an export is baked, and VRoid's
   face sliders (eye size and shape, chin, face length) exist only in VRoid Studio's editor.
2. **The hair's crown and back from 60-120 deg** (`vroid_face.jpg` 60 / 90, `vroid_whatif_hair.jpg`): the sample's back cards
   converge on a whorl at the back of the crown (built for the twin tails), read as a bun in profile; the drawings' crown is
   smooth to the back. Shino's hair (the what-if) is smooth there but has a blunt hime fringe, not the drawn pointed one.
3. **The side locks hide the face in profile** (`vroid_face.jpg`, the profile framing): the lengthened face-framing locks
   hang over the cheek; the drawing keeps one narrow lock in front of the ear.
4. **The hair's painted light is faint**: B2e's lobe is on, but the concept's ash-pale outer hair on her right barely shows
   (`vroid_vs_concept.jpg`); it needs B2e's per-lock share, which VRoid's cards do not carry.
5. **The latex**: wet, with crisp streaks on the rounder forms (once `toonlook.prepare` computed the latex's curvature --
   the first pictures skipped it and the streaks were gated to nothing); still fewer and smaller highlights than the concept's
   bust and arms, and the piping only on the cut edges (no princess seams).
6. **A lip highlight reads as a fang** in the concept framing (the VRoid face texture's lower-lip gloss).
7. **Hands**: VRoid's simple hands, fingers relaxed by a fixed curl; fine at full figure, plain in close-up.
