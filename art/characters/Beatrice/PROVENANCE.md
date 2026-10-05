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

## B1e: the pelvis, second round (2026-10-05, answers `previews/review_stage4.md`) — in progress

- Reference added: `good-body-silhouette.jfif` (683 x 1024, supplied by Erik 2026-10-05; front view, head to upper
  thighs, the earlier high-cut suit; "a good body shape, the same as the original image"). SHA-256
  `f1fd420732ad1fa18dab435340409632515b8d52a74a266510e97b93b9a8de43`. Body SHAPE only; nothing of its suit is used.
- Step 1, the gap between the legs: the pelvis is ONE surface across the mid-plane, the smooth union of the
  half-body and its mirror image (`charkit/bodysuit.py::MidUnion`, table `MIDPLANE`), replacing the mid-plane cut
  (`weld_rows`) + `midline_fillet` from under the crotch to the waist with one mechanism on the loft's own grid;
  fuller upper inner thighs (`PELVIS["inner"]`): a slot about 12 mm wide for 6 cm under the crotch, a round arch.
  Stage 4's pelvis is kept as the shape `stage4` (the hip pictures' "before").
