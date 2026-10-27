# Whole octopus — 2026-10-27

Built on the one-arm proof below; its files are unchanged and still reproduce
it. Open [index.html](index.html) for the pictures and
[CONTROLS.md](CONTROLS.md) for the per-limb control layout. This is a
HUMAN-TEST handoff: render-only, nothing pushed or merged, no game or
simulation file touched.

## What it is

- `scripts/octopus_math.py` holds the controls, `pose_octopus`, the body/crown
  geometry, the skin weights and the demo cycles. It is pure NumPy and imports
  `arm_math`:
  - the hero arm IS `arm_math.geometry()` cut at junction ring 20 (s = 0.156);
  - the taper is `arm_math.radius`, the per-vertex blend is `weights_at`'s
    two-joint rule, and the runtime quaternion is `arm_math.quaternion`.
- **Junction.** The body, crown and web are one structured grid, with two
  sheets (dorsal and oral) whose boundary is exactly the eight arm junction
  rings plus the web rims. Those ring vertices are shared, so the whole
  surface is one closed manifold with no cut cylinders. Grid vertices are
  ray-cast onto an implicit smooth union (mantle, head, eye bumps, eight arm
  roots, a scalloped web slab, an oral disc). Near each ring a "collar" makes
  the crown follow the tube's own surface lines (G1 on the crest), a seam snap
  removes the blend's inflation, and a short relaxation removes shear folds.
- **Rig form.** A bone chain per arm, as in the proof. Hero: 33 joints per arm.
  Game: every 2nd joint of the SAME curve (17 per arm), plus 6 body bones
  (body, mantle, three translation-only squeeze walls, siphon). The pose is a
  discrete fixed-length rod: per joint a bend increment about a material axis
  set by `bend_dir`, plus twist; root lift and swing are bent in over the first
  6 joints. A chain was chosen over "fewer bones + curve in the skin" because
  the game skins on the CPU through raylib bones only: a curve-in-skin path
  would mean a parallel skinning path, which the renderer seam forbids. Halving
  the joints costs at most 12.3 mm against the hero rig across random
  full-range poses.
- **Game LOD.** 12-sector arm tubes and a coarser crown, with no modelled cups.
  The suckers, mottling and eyes are baked from the hero in Blender
  (selected-to-active: diffuse colour, tangent normal, roughness) into a
  2048² albedo, with gloss = 1 − roughness/0.65 in ALPHA (skin ≈ 0.32, cups
  ≈ 0.45, eye ≈ 0.9), spliced in by `charkit.glbtex.replace_image`, plus the
  normal map.

## Numbers (prototype/octopus/*.json; default size `mid` since #106)

| | hero (.blend) | game (.glb) |
|---|---:|---:|
| triangles | 97,952 (132,512 at `proof` size) | **12,200** |
| vertices | 49,656 | 6,108 authored / 8,502 loaded |
| joints | 270 | **142** |
| modelled suckers | 336 (624 at `proof` size) | 0 (baked) |

Runtime, measured through the loaded GLB and raylib's native CPU skinning
(`runtime_octopus.py --benchmark`, 120 samples, methodology as for the arm).
These are CPU timings, NOT whole-game FPS:

| CPU per frame | 1 octopus median (p95) | 4 octopi median |
|---|---:|---:|
| controls + `pose_octopus` | 3.21 ms (4.16) | 12.9 ms |
| write 142 joint transforms | 0.73 ms | 2.92 ms |
| raylib skin + upload | 0.17 ms | 0.68 ms |
| **total** | **4.12 ms (5.39)** | **16.6 ms** |
| total, two front arms stretching | 4.17 ms (4.55) | 16.9 ms |

(Before #106: 4.01 ms (4.28) / 15.7 ms; the stretch adds the scale output and,
for a stretched arm, the compensated translation and the scale write.)

The skinning itself is cheap. The cost is the pure-Python pose and write; the
first optimisation would be to batch the 8 arm recursions, since a scalar loop
per arm is not needed.

Loaded raylib skin vs the NumPy reference: ≤ 11.4 µm over 54 poses, 18 of them
with stretched arms (gate 10 µm times the size's scale, 19.7 µm: float32 error grows
with coordinate size; it was 5.7 µm at scale 1). Blender's evaluated hero skin vs the reference: ≤ 1.4 µm.

Property checks (`verify_octopus.py`, report in
`prototype/octopus/property_report.json`):

- out-of-range controls are refused;
- the pose is deterministic and every frame is an exact rotation;
- segment length error ≤ 4.7e-16 m (300 random poses), and game joints are
  bit-identical to the even hero joints;
- arm independence: in 60 trials, moving arm j never moves arm i, and
  mantle/siphon controls never move an arm;
- continuity: sweeping every control across its full range in 400 steps, the
  worst max/median step is 2.10 and the worst frame step is 0.032 rad;
  `bend_dir` ±1 give the same pose;
- both meshes are closed manifolds, the junction rings are stitched, weights
  are normalised with ≤ 4 influences, and the bind skin reproduces the mesh.

Deformation:

- **Shown poses:** arm triangle-area ratio ≥ 0.84, crown ≥ 0.029, at most 8
  inverted slivers out of 12k.
- **Random full-range poses:** arm tubes ≥ 0.58, hero arm sections 0.95–1.00,
  but the crown/web is ≥ 0.003 with up to 72 inverted slivers (a known
  limit, see CONTROLS.md).

## Rebuild

```powershell
$o = 'C:/Users/steen/projects/breach/.claude/worktrees/33-octopus-concept/art/characters/octopus'
$breach = 'C:/Users/steen/projects/breach'      # read-only root, main at cfcba40
$py = 'C:/Users/steen/anaconda3/python.exe'
$blender = 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe'
& $py "$o/scripts/verify_octopus.py"
& $blender -b --factory-startup -P "$o/scripts/build_octopus.py" -- --root $breach --stage all
& $py "$o/scripts/package_octopus.py"
& $py "$o/scripts/runtime_octopus.py" --root $breach --benchmark
& $py "$o/scripts/runtime_octopus.py" --root $breach --shots "$o/previews/octopus/game/game" --frames 80,170 --focus 20:28:0 -- --level playground
& $py "$o/scripts/runtime_octopus.py" --root $breach -- --level playground
```

The steps, in order:

1. `verify_octopus.py`: the properties, about 2.5 min.
2. `build_octopus.py --stage all`: the .blend, the bake, the GLB and the
   renders, about 6 min on the GPU. `--stage swim` re-renders only the swim
   stills and the GIF frames.
3. `package_octopus.py`: the GIF and the comparison sheets.
4. `runtime_octopus.py --benchmark`: parity and timings.
5. `runtime_octopus.py --shots`: the game screenshots.
6. `runtime_octopus.py` with no flags: play it, with live posing.

`--shots` needs an ABSOLUTE prefix: the tool changes into the root checkout
before main.py runs.

## Body plan (#106): size, arm/body ratio, stretch

Size and proportions are now dials, one row of `SIZES` in
`scripts/octopus_math.py` (plain-language table in [CONTROLS.md](CONTROLS.md)):
`body_length_m`, `arm_body_ratio` (rest arm, crown root to tip, over the body
length; the arm keeps the crown's girth) and `stretch_max`. `OCTOPUS_SCALE` is
now derived (`body_length_m / 0.71`). Switch with `om.set_size(name)` or
`OCTOPUS_SIZE=<row>` for any build or tool.

**Erik's ruling, 2026-10-28:** option A's proportion on bigger bodies. Every
family row has `arm_body_ratio` = `ARM_BODY_RATIO` = 1.8 and `stretch_max` =
`FAMILY_STRETCH_MAX` = 1.40 (free arm; whole arm x1.31), so the rows are one
shape at three scales (`ARM_LAMBDA` 1.085 on all three). The fixed 2.0 / 2.5
marine span is dropped. Rows: `mid` (the default, medium, body 1.4 m: arm
2.52 m, stretched 3.31 m, span 5.34 / 6.91 m), `small_preview` (option A's
1.0 m body: arm 1.80 / 2.36 m, span 3.82 / 4.93 m), `big` (option C's 1.885 m
body, a preview of a later larger size: arm 3.39 / 4.45 m, span 7.19 / 9.30 m),
and `proof` (the pre-#106 animal at scale 1, kept rebuildable). The #106
options (`mid_bulk`, the 1.24-ratio `mid`, `mid_heavy`) are at 65df56a.
`verify_octopus.check_sizes` gates the ratio, the shared stretch fraction and
mid's 3.2-3.4 m reach.

**Arm length law.** Per hero segment a length factor, exactly 1 for the five
segments inside the crown and ramped in (smoothstep) over the three segments
past the junction ring, so the junction never moves and the taper has no kink.
The rest length (`ARM_LAMBDA`, solved from the row) is baked into the bind mesh
(`bind_x`, piecewise linear with a knot at each hero joint, so the proof's
two-joint weights stay exact); the hero keeps only the sucker stations whose
rest spacing meets the proof's own spacing rule (option B: 97,952 hero
triangles instead of 132,512; the game mesh is unchanged at 12,200).

**Stretch** (7th arm control, 0..1). The same ramp carries the elongation
`1 + stretch (stretch_max - 1)`; each joint gets a UNIFORM scale
`1 / sqrt(elongation of its incoming segment)` (volume conserving: measured arm
volume +0.9 % at 0.5, +1.7 % at 1). Joints 0..6 always stay at scale 1, so the
crown is bit-identical under any stretch.

**In the game.** raylib 5.5.0.4's CPU skin applies a frame pose's scale after
rotation and translation, in model axes about the model origin (measured: a
bone scaled to 0.5 threw its vertices 0.33 m). For a uniform scale that is
still exact when the translation is pre-compensated, which
`runtime_octopus.LiveOctopus.write_pose` does (`p / s + (s - 1) R b`). Loaded
raylib skin vs the NumPy reference with stretched poses: <= 5.7 um; the same
check without the compensation is off by 206 mm and without the scale by 9 mm,
so the gate can fail. Blender honours the uniform bone scale natively (hero
parity 2.4 um incl. the stretched reach).

```powershell
& $py "$o/scripts/verify_octopus.py"                                                  # also checks every SIZES row
& $blender -b --factory-startup -P "$o/scripts/build_bodyplan.py" -- --root $breach   # the three sizes + strip
& $py "$o/scripts/package_bodyplan.py"                                                # labelled sheets
& $py "$o/scripts/runtime_octopus.py" --root $breach --mix 0 --offset 14,-3 --pose stretch --shots "$o/previews/bodyplan/game/mix" --frames 60,105 --focus 30:24:17,43 -- --level playground
```

(`--pose rest` / `--pose reach` give the two stills in `game/`.) `--mix N`
draws only the Nth player as the octopus (the rest keep their own
looks, the marines) and `--offset` moves where it is DRAWN, render-side only.
Pictures: `previews/bodyplan/` (`options_top.jpg`, `options_34.jpg`,
`<row>_top.jpg`, `<row>_34.jpg`, `stretch_strip.jpg`, `game/`). In the game
the default octopus at rest measures about 11.7 tiles across with its curled
rest arms (12 straight), beside marines drawn 6 tiles tall.

**Found while doing this, not fixed (other systems):**
- raylib 5.5.0.4's CPU skin gets the NORMALS of every posed limb wrong: a pure
  bone translation of 1 m turns them by up to ~89 degrees and scales them 0.03x to 2x
  (measured on `animNormals`), and `marine_shader` lights with them. It
  predates #106 (an unstretched curl already deviates by up to 95 degrees on
  that arm) and touches every skinned look; stretch makes the octopus's
  worse. Not visible at the playground's light level in the shots taken.
- `swing`'s sign was documented backwards: positive turns an arm toward the
  BACK on both sides. The control is unchanged; CONTROLS.md now says so.
- `verify_octopus.py` compared posed (scaled) areas against unscaled rest
  areas; harmless at scale 1, wrong at any other. Fixed.

**Changed expectations, on purpose:** the segment-length check is now
"equals the length law" (it was "equals STEP"), the hero section-area check is
measured against the thinning law's expectation, and the raylib parity gate
scales with the size.

## Patterns (#113)

Seven skins, one PATTERN TABLE (`scripts/pattern_math.py::PATTERNS`): the
default plus six taken from real animals. A row holds the animal, its colours
(sRGB 0-255) and sizes (metres on this 1.2 m-arm animal); its `builder` names a
pattern function in `scripts/build_patterns.py`. The `default` row IS
`build_octopus.skin_material` / `cup_material`, unchanged.

| row | animal | look |
|---|---|---|
| `default` | common octopus (the build's own skin) | red-brown mottle |
| `zebra` | mimic octopus, wunderpus | cream, dark bands ringing the arms and tightening toward the tips, warped chevrons on the mantle |
| `reticulated` | honeycomb moray, giraffe | dark polygons in a thin pale net, shrinking along the arms |
| `rosette` | leopard, jaguar | gold-ochre, broken dark rings round a darker centre, solid spots between them and down the arms |
| `red_whitespot` | *Callistoctopus macropus* | brick red, scattered white spots of uneven size, oval on the arms |
| `blue_ringed` | *Hapalochlaena lunulata* | ochre-yellow, brown blotches each carrying a black-edged electric-blue ring |
| `ghost` | Casper octopus, dumbo octopuses | pale white with a pink blush and faint specks; subsurface in the hero only |

**Coordinates.** Every pattern is drawn in rest-pose coordinates stored per
vertex (`pattern_math.skin_coords`, written by `build_patterns.py` as point
attributes on the hero mesh), so it rides on the skin and stretches with it:
`pat_body` (rest position), `pat_arm` (the arm's length coordinate
`U(x) = ∫dx / (r_ref (r/r_ref)^0.7)` and the angle around the arm, per arm: a
feature shrinks with the radius toward the tip), `pat_w` (morphs the body field
into the arm field over the proximal arm, s 0.156 to 0.246, so the junction has
no seam) and `pat_s`. Countershading: the existing `pale` attribute mixes 85 %
toward the row's oral colour; suckers take the row's rim/bowl colours. Geometry,
rig, controls and the committed hero .blend are not touched.

**Game.** Each variant's albedo is baked onto the SAME game mesh with the
build's own bake (`build_octopus.bake_pass`), and the committed gloss alpha is
kept: `prototype/octopus/variants/octopus_albedo_gloss_<name>.png` (committed,
1.1 to 1.9 MB each). `package_patterns.py` makes
`prototype/octopus/variants/octopus_game_<name>.glb` = the committed GLB with
only the baseColor image swapped, and checks every other buffer view (333) is
byte-identical. The GLBs are ~8 MB each and gitignored: rebuild them with the
second command below.

```powershell
& $blender -b --factory-startup -P "$o/scripts/build_patterns.py"           # all rows, ~3 min; --only zebra,ghost
& $py "$o/scripts/package_patterns.py" --root $breach                        # GLBs + lineups, seconds
& $py "$o/scripts/runtime_octopus.py" --root $breach --model "$o/prototype/octopus/variants/octopus_game_zebra.glb" -- --level playground
```

`build_patterns.py --save-blend` also writes a per-variant hero .blend
(gitignored). Pictures: `previews/patterns/` (`<name>_top.jpg`,
`<name>_34.jpg`, `game_<name>_top.jpg`, `lineup_top.jpg`, `lineup_34.jpg`,
`game_lineup.jpg`, `game/zebra_0080.png`, `game/blue_ringed_0080.png`).

**Default unchanged** (`prototype/octopus/variants/patterns_report.json`): the
default row re-baked through this path differs from the committed albedo by
1/255 on 7 of 4.2 M texels, its re-baked gloss is identical, and a full
`build_octopus.py --stage assets` after the `bake_pass` refactor gave the same
GLB mesh/skin/animation/JSON digests, an identical geometry .npz and PNGs
within 1/255 on 8 (albedo) and 5 (normal) texels (GPU bake noise).

**Weak, per pattern.**
- `rosette`: the weakest. It reads as spotted-cat but the rosettes are small
  and sparse beside a real leopard's packed coat, and the gold base is flat.
- `zebra`: the mantle chevrons are blotchier than a mimic's clean bands; the
  arm bands are very regular.
- `reticulated`: the net gets thick relative to the patches on the thin distal
  arms.
- `blue_ringed`: real rings are iridescent and flash; here they are flat
  albedo, and in the game bake they read darker.
- `ghost`: under the studio key the mantle top blows out to near-white; the
  translucency is a hero-only subsurface, the game is a pale albedo.
- `red_whitespot`: good; the spots are uniformly white with no pinkish rim.

## Weak, bluntly

- **The arms read too thin and too straight next to the concept.** The sheet's
  arms are about twice as thick at the base, with strongly hooked tips. The
  1.2 m proof arm was kept as instructed. An arm-girth multiplier for the
  octopus, and a stronger rest `tip_curl`, would be the first look fix
  (Erik's call).
- The mottling is uniform and low in contrast, with no papillae or pale
  spots; the concept is darker and richer. The underside is close to white.
- **Eyes** are glass beads in small bumps, not turrets with lids. The
  **siphon** exists and is rigged, but it is hidden by the arm-root blend in
  every view. The mouth is a dark dot with no lip or beak.
- **Junction.** It is continuous and the webbing reads, but a faint crease
  remains where the crest collar hands over to the web side columns (visible
  in the close-up, right arm). Folds appear in extreme root pitch.
- Everything lies flat: the side profile does not stand on its arms like the
  sheet's.
- **No collision of any kind.** Arms pass through the floor in swim and walk,
  and through each other.
- **The GLB is 8.3 MB.** The normal map is exported as a 16-bit PNG; making it
  8-bit would roughly halve that.
- The game preview draws every unit as an octopus (use `--mix` for one
  octopus among the marines). The size is now the SIZES row (#106). The demo loop switches from walk to swim without a blend.

## Systems

Reused: `UnitModelRenderer` (live posing through its CPU-skin seam),
`marine_shader`, `lit3d`, projected shadows, `charkit.glbtex.replace_image`,
`charkit/preview_in_game.py` and `tools/e2e_drive.py` (read-only from root),
and `arm_math` (the proof's arm, taper, blend and quaternion). New and
prototype-local: `octopus_math` (the octopus controller and its geometry) and
its skeleton, a new body plan as approved by
docs/reference/adding_character_models.md §1. No renderer, shader, simulation,
config or production file changed.

---

# One-arm feasibility proof — 2026-10-08

Open [index.html](index.html) for the local animation player, three posed views,
dorsal view, game screenshots and downloads. The preserved original concept is
[concepts/octopus_concept_v1.png](concepts/octopus_concept_v1.png).

The proof works: one ordinary weighted mesh bends from extended to a strong
three-dimensional curl and into a scripted tip contact. The same pure curve
controller supplies Blender bone poses and newly evaluated poses in Breach's
actual `UnitModelRenderer`. This is a HUMAN-TEST handoff, with no push, merge or
gameplay activation. The parent reviewed the initial renders and accepted the
single correction round: clearer curl framing, oral/dorsal contrast, visible
contact tip. Material and cup detail are substantially below concept polish.

## Inspect and edit

- [prototype/octopus_arm.blend](prototype/octopus_arm.blend): editable mesh,
  closed cup islands, UVs, material, 33-bone chain, normalized vertex groups,
  linear armature modifier and editable nine-second inspection animation.
  Opens on the curl at frame 73; play frames 1–216 at 24 fps. Sources are also
  embedded as Blender text blocks. No startup script or external constraint is
  needed to open/play/edit the file.
- [prototype/octopus_arm.glb](prototype/octopus_arm.glb): 899 KB portable skin
  with one tiny, static `Idle_Loop` clip to satisfy the existing game's loader.
  It intentionally does not contain the controller as an exported constraint.
- [scripts/arm_math.py](scripts/arm_math.py): palette, taper, geometry, curve
  angles, twist and independent reference skinning. `pose(curl, contact)` accepts
  nonnegative controls whose sum is at most one. `cycle_weights(seconds)` is
  the smooth repeating demonstration.
- [previews/arm/arm_loop.gif](previews/arm/arm_loop.gif): actual mesh renders,
  54 frames over nine seconds. The HTML scrubber uses the same JPEG frames.
  It is a rendered review player; the Python game preview evaluates live poses.

The specimen is displayed **oral-side up** so the suckers remain easy to inspect.
That is a presentation orientation, not an octopus locomotion decision. In the
dorsal still the same specimen is turned over. The base has a closed cut face
because it is an isolated arm, not a finished body junction.

## Reproduce on this machine

PowerShell, from any folder:

```powershell
$arm = 'C:/Users/steen/projects/breach/.claude/worktrees/33-octopus-concept/art/characters/octopus'
$breach = 'C:/Users/steen/projects/breach'
$breachPy = 'C:/Users/steen/anaconda3/python.exe'
$blender = 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe'

# Geometry/pose/contact properties, independent of Blender and raylib.
& $breachPy "$arm/scripts/verify_arm.py"

# Rebuild the editable file, GLB, stills and the short render sequence.
& $blender -b --factory-startup -P "$arm/scripts/build_arm.py" -- --root $breach --animation
& $breachPy "$arm/scripts/package_previews.py"

# Rebuild only editable/exported assets (no renders).
& $blender -b --factory-startup -P "$arm/scripts/build_arm.py" -- --root $breach --assets-only

# Interactive actual-game preview: continuous live curve-to-bone posing.
& $breachPy "$arm/scripts/runtime_arm.py" --root $breach -- --level playground

# Fixed pose choices also available: --pose extended, --pose curl, --pose contact.
# Hidden E2E capture: actual main.py, normal light field/camera/shader/shadows.
& $breachPy "$arm/scripts/runtime_arm.py" --root $breach --shots "$arm/previews/arm/game" --frames 90,180,270 -- --level playground

# Hidden context: loaded GLB skin parity and CPU timings for 1 / 8 arms.
& $breachPy "$arm/scripts/runtime_arm.py" --root $breach --benchmark --samples 120
```

Dependencies: Blender 4.5 (its bundled NumPy), Breach's Python with NumPy and
pyray, plus Pillow for GIF/JPEG packaging. No new packages were installed.
The machine's older SciPy is incompatible with its NumPy; this proof does not
use SciPy. The root checkout is read-only input at
`2c560cdfb0a9808c037b6d464c91140c25daa7cd` (`fire-12`). It contains the canonical
`charkit.glbtex`, `preview_in_game.py`, `tools/e2e_drive.py`, renderer and working
game dependencies absent from this main-based art branch. Moving the art folder
is supported; supply the current root explicitly. There is no implicit import
of the entire fire-12 branch.

## Construction and export seam

The 1.20 m centreline contains 32 equal segments and a terminal joint: 33 joints
total. Unit tangent samples are integrated at fixed segment length. An oral
axis follows the shortest rotation between adjacent tangents, then receives
authored twist; there is no Frenet normal that can flip at an inflection.
Both heading and elevation vary, so the curl/reach includes out-of-plane bending
and twist. Smooth control interpolation provides a continuous loop with holds.
This is elementary vector arithmetic, not a physical soft-body model.

The skin is a tapered tube with 24 vertices per ring and 128 nonzero rings plus
closed base/tip caps. There are two rows of 40 closed hollow cups, progressively
smaller and closer toward the tip. The cups' weights come from their attachment
station, keeping each lip together. Skinning uses two adjacent joints and
ordinary linear blending, with no scale animation or volume correction.

One mesh, one material, 8,034 authored vertices, **15,744 triangles**. UV seams
and normal splits produce 13,010 loaded GLB vertices. Eight independent arms
would total 125,952 triangles and 264 joints. The mesh/material can be shared,
but each arm needs its own newly evaluated pose; neither complete-animal anatomy
nor inter-arm placement has been built. This budget is a prototype measurement,
not a production allocation or size ruling.

The runtime tool replaces the renderer module's raylib reference with a
process-local forwarding object. Only `update_model_animation` for the loaded
prototype gets a fresh one-frame `ModelAnimation` argument; it then calls the
original native raylib function. Existing `UnitModelRenderer._draw_one`,
`marine_shader`, `lit3d`, light-field sampling and projected-shadow draw stay in
charge. The installed pyray package and production source/config are untouched.
No poses are written to `Unit` or any simulation field.

The GLB's static clip is only a loader compatibility token. The game capture
completed **275 frames, exit 0, 2,475 procedural skin calls and 198 distinct
control pairs**. Captures at frames 90 / 180 / 270 show curl / reach / extended;
they are not clip playback. All units in that preview process use the prototype
at their original game positions/facings. The frame-180 game view has no contact
prop; the studio contact picture is the geometric-contact evidence.

## Measurements

Recorded on AMD Ryzen 7 3700X / RTX 3070, Windows 10, Python 3.11.7,
pyray 5.5.0.4. [runtime_report.json](prototype/runtime_report.json) records the
environment and exact measurements. 20 warmup updates, then 120 batches of
sequential independently phased arm updates on one shared loaded model, matching
the current renderer's model-sharing seam. Wall time uses `perf_counter_ns`.

| CPU operation | 1 arm median | 8 arms median |
|---|---:|---:|
| Curve evaluation / frame transport | 0.448 ms | 3.601 ms |
| Write joint transforms | 0.157 ms | 1.352 ms |
| Native raylib skin + buffer submission | 0.204 ms | 1.640 ms |
| Total, including control evaluation | **0.827 ms** | **6.707 ms** |
| Total p95 | 0.988 ms | 7.185 ms |

These are CPU submission timings, **not whole-game FPS**. They exclude draw,
shader work, shadow draws, GPU completion, simulation and other actors. The
renderer uploads skin buffers per posed draw. Eight-arm use remains a meaningful
CPU budget; future performance work should be measured in the intended scene.

[property_report.json](prototype/property_report.json) checks 181 intermediate
loop poses against the brief's properties. Changes to topology, weights, scale,
curve continuity or excessive bend must break the relevant assertion:

- Closed geometry and finite vertices; nonnegative normalized weights; bind
  skin equals source geometry. Fixed bone-segment length error ≤ 1.2e-16 m.
- Right-handed orthonormal frames, repeated evaluations identical, and no
  discontinuity across the loop boundary. Largest joint-frame step at 20 Hz:
  0.1822 rad, under the 0.19 rad demonstration threshold.
- Section areas measured from **deformed ring polygons**, not centreline length:
  minimum 99.2474% of rest, maximum 100.0000% (97% / 100.1% thresholds).
  The worst actual triangle retains 76.58% of rest area (52% threshold), so the
  demonstrated concave bend has no collapsing triangles.
- Conservative bounding spheres made from **deformed cup vertices** remain
  separated by at least 2.746 mm. This proves cup-to-cup clearance for the sampled
  poses, without pretending to be a general collision solver.
- Contact endpoint `(1.0, 0.15, 0.5006524)` m touches the fixed `x=1.0` plane;
  tip error < 1e-12 m, maximum penetration of **any skin/cup vertex** is zero.
  The plane's flat central face covers that point in the studio render.
- Blender's evaluated armature mesh matches the shared reference within
  0.423 μm across the still poses. Loaded raylib skin matches it within 1.212 μm
  over 49 loop samples (10 μm gate); GLB rest vertices differ by < 0.060 μm.
  Raylib parity therefore also checks axis conversion, exported weights, bind
  orientation and the live transform adapter.

No production source changed, so no simulation/golden baseline was regenerated.
The scoped checks are geometry properties, evaluated Blender parity, loaded-GLB
native skin parity and the clean real-game E2E run.

## Limits and skill lessons

Only the shown curve family is assessed. No general body self-intersection,
inter-arm collision, obstacle avoidance, adhesion, suction mechanics, grasping,
contact-force solver, AI or navigation is implemented. Cup sphere separation
does not prove cup-versus-body clearance away from its own attachment. Contact
is an authored geometric endpoint; it is not robust wall collision during a
transition. The full loop hides the slab, since other poses can pass beyond it.
Render-only float poses make no promise about deterministic gameplay.

No new image generation was used for proof images. All meshes, atlas pixels and
poses are original procedural prototype data; the original concept remains
unchanged. No third-party mesh or animation was incorporated.

2026-10-08 scripted-character lesson, recorded here per the brief: the humanoid
fusion/retopo/46-clip pipeline was deliberately skipped for a single non-humanoid
deformation experiment. A tiny rest clip can satisfy the existing loader while
the renderer accepts fresh curve-driven transforms at its normal skinning seam.
Exported skin must be compared on loaded vertices: Blender's and raylib's axis/
bind conventions otherwise hide mistakes. Blender treats image alpha as coverage
but Breach treats it as gloss, so use opaque studio RGB and the canonical
`charkit.glbtex.replace_image` to preserve gloss in the game GLB. Tiny per-joint
NumPy operations and pyray struct constructors dominated CPU time; scalar
arithmetic/direct writes reduced the measured eight-arm update from ~60 ms to
~6.7 ms without changing the skin/controller design.

## Systems

Existing systems reused: `UnitModelRenderer`, `marine_shader`, `lit3d`, projected
unit shadows, charkit's read-only texture replacement and preview helper, and
the existing E2E driver. No renderer, shader, lighting, physics or simulation
system was added.

Prototype-local reusable component: `scripts/arm_math.py` plus its ordinary
33-joint mesh. Any extension of **this experiment** should use that controller
for Blender and runtime together, and re-run loaded-skin parity. This is not a
second production character/skeleton pipeline or an approved whole-animal rig.
