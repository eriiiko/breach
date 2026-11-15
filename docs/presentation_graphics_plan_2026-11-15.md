# Presentation: the graphics plan

2026-11-15. Erik handed the graphics for the promo video to Claude ("the
whole graphics stuff, could you plan it, can I just leave it in your hands?").
This is the plan: what gets built, in which order, and what each piece is
judged by. Milestone 24, tracker #207. The script it serves:
`docs/promo_video_script_2026-11-15.md`.

Written on a day when the PC was busy, so nothing here was rendered or
measured. What is said about today's water and light comes from reading the
code and the rules; each such claim is marked "read, not seen".

## 1. Who decides what

- **Claude**: which techniques, the order, the designs, the patches, the
  tuning up to a first good look. Each track ends with before and after look
  shots.
- **Erik**: what the game should look like when two good options differ in
  taste, anything that changes the simulation or the gameplay, and the story.
  Erik looks at the look shots when he wants to; a play test happens when he
  asks for one.
- Each track is an arc as usual: design doc with its Systems section, a
  critique, patches with gates. All of it is render-only unless a section
  says otherwise, so the determinism goldens are not touched.

## 2. The tracks and their order

| Order | Track | Issue | Why here |
|---|---|---|---|
| 1 | Marks | #180 | Smallest design, shows in almost every shot, and the game keeps it |
| 2 | Water look | #97 | Research is done (section 4); independent of everything else |
| 3 | 3D walls, with the camera tilt | #209, G6 in #93 | Changes every picture; the departments are dressed on top of it |
| 4 | Capture: director camera, frame export | #208 | Needed before any shot can be tried; small |
| 5 | Video-only looks | #212 | Each is small and waits for a scene to exist |
| 6 | Coloured bounce | #213 | One shot; rides on the marks |
| last | Roof lamps and more shadows | none | Decided once walls and tilt exist (Erik delegated this) |

Tracks 1, 2 and 4 do not touch each other and can run in any order or side by
side when the budget allows. Track 3 should not start before the residency
arc's render-facing patches are quiet, since both touch the renderer's frame.

Not graphics, running beside this: departments and props (#210), cast (#211),
sound and music (#214).

## 3. Marks (#180, with #213)

**What**: one layer that remembers what happened in a match and draws it on
the floor and the walls for as long as the match lasts: bullet holes, blood,
scorch, ash, settled debris, spent magazines, bodies and parts of bodies.

**How** (the issue has the first sketch; this adds today's items):
- One image over the level, drawn in the lit floor pass through the shared
  light code, so marks are lit and shadowed like the floor they lie on. A
  second, smaller set for wall faces once walls are 3D.
- A single `paint` door. Everything that leaves a mark calls it with a place,
  a kind and a direction. The callers already exist as events: a hit and its
  direction (#185), a tracer stopped by a wall, a blast and its reach, a
  reload, a kill.
- Marks are pictures from a small atlas per kind, turned and scaled by a
  keyed draw so a replay paints the same marks.
- Things with a shape (a magazine, a limb) are small models through the
  static prop path while they are few, and are baked into the image when they
  are many.
- Blood knows its direction: a spray on the wall behind the one who was hit,
  a pool that grows for a few seconds under a body.
- Render-only. Nothing in the simulation reads a mark.

**Coloured bounce (#213)**: read, not seen: the light sweep carries light
from sources and absorbs it; it does not give light back from a lit surface,
and a mark is unknown to it. Plan: first check this on a free machine. If it
holds, a mark that is strongly lit and strongly coloured becomes a weak light
of its colour for that frame, handed to the one light assembly. Its strength
follows the light falling on it, so it exists only under a beam.

**Judged by**: new look shots `marks_firefight` (after a recorded fight) and
`blood_white_wall` (the script's shot 2.3).

## 4. Water (#97)

Erik, 2026-11-15: the simulation works; the picture on top should be made
impressive. The foam in particular is not convincing. Research how.

### 4.1 What is there today (read, not seen)

`shaders/water.fs` is one pass that reads the ripple height and the water
depth and does refraction of the floor, a depth tint, glints from the lights,
caustics, and foam. The design is sound (`docs/architecture/graphics/water_rendering.md`).
Three things hold the look back:

1. **Everything is as coarse as a tile.** Ripple and depth are one value per
   tile, and at play zoom a tile is tens of pixels wide. No wave can be
   smaller than a tile, which is why they read too big (#97). The only finer
   detail is three fixed sine waves, which read as a pattern.
2. **Foam is a white tint with no memory.** Where the ripple is steep or the
   depth changes fast, the pixel is mixed toward white. It has no texture,
   appears and vanishes in one frame, has the shape of the tiles, and does not
   travel with the water.
3. **The water's flow is not used.** The simulation knows how fast and which
   way the water moves on every tile (`flow_vx`, `flow_vy`); the pass does not
   receive it. Nothing on the surface drifts.

### 4.2 What the research says

- **Flow maps** (Vlachos, Valve, 2010; Left 4 Dead 2 and Portal 2). A small
  repeating picture of fine ripples is slid along the local flow direction.
  Sliding alone stretches it, so two copies are slid half a cycle apart and
  cross-faded; the eye sees water that streams and never repeats. Valve drew
  their flow by hand. We have it from the simulation, per tick, for free.
- **Small waves carried by the flow** (Chentanez and Müller, 2010). A coarse
  height-field simulation, like ours, gets fine waves added in the picture and
  carried along by the simulated velocity. Same idea, and the paper is the
  reference for doing it on a simulation.
- **Foam as an amount that is made, carried and lost** (Ojeda and Susín;
  Grenier's river editor; Unity's water system; Mario Kart water breakdowns).
  All of them keep a foam amount per place. It is added where water is
  violent, moved with the flow every frame, and fades over seconds. The
  picture is that amount applied to a foam texture.
- **Foam that breaks up, never fades** (Parberry's sea foam). Real foam does
  not turn transparent; its bubbles pop and holes open. So the amount is used
  as a threshold on a bubble pattern: much foam is a closed white sheet, less
  is lace, little is a few strings. This is the single thing that separates
  foam that convinces from foam that looks like fog.
- **Caustics from the surface's shape** (Wallace, 2011). Bright lines on the
  floor come from how much the surface squeezes the light together: follow
  each bit of light through the surface to the floor and compare the area it
  covers before and after. We already do a soft form of this from the coarse
  ripple; done on the fine surface it gives the sharp moving net of light.
- **Spray** (Chentanez and Müller). Where the water is too violent for a
  surface, particles are thrown and fall back as foam.

### 4.3 The plan, in steps

Each step is one patch with look shots before and after. All render-only; the
water simulation is not changed.

- **W1. A fine surface that flows.** The flow velocity is uploaded beside
  ripple and depth. The three sine waves are replaced by a fine ripple
  texture, slid along the flow in the two-copy way, its strength growing with
  the flow's speed and the local agitation. Still water becomes nearly flat
  with a slow drift, so a flashlight gives one clean reflection that breaks up
  when something disturbs it; running water streams. This also answers #97:
  wave size becomes a dial in metres, independent of the tile.
- **W2. Foam with a life.** A foam amount kept in its own image, finer than
  the tiles, updated every frame: added where the surface is steep, where
  fast water meets a wall or runs shallow, at the advancing edge of a flood,
  where water pours in, and where something hits the water; carried by the
  flow; fading over a few seconds. Drawn as a threshold on a bubble texture at
  two sizes, itself sliding with the flow, lit by the room's light, with a
  soft shadow on the floor beneath. Thick foam hides the floor; thin foam is
  lace.
- **W3. Things in the water.** A unit walking through water leaves a wake and
  a little foam; a bullet throws a small splash; a blast throws a ring of
  spray that falls back as foam. Through the one particle system.
- **W4. Sharp caustics.** The caustics are recomputed from the fine surface
  of W1 with the area method, scaled by depth and by the light that is there.
  Under a flashlight in a flooded corridor the floor carries a moving net of
  light.
- **W5. Wet floor.** Where water has been, the floor stays dark and glossy
  for a while and dries from the edges. Kept in the marks layer of section 3.
  Blood in water tints it. Last, and only if the earlier steps leave time.
- Also fixed on the way: the water receives the smoke's shadow (a finding
  from G2, in #97).

W1 and W2 are most of the gain. W2 is the answer to the foam.

**Judged by**: the existing `steam_over_pool` shot, and new shots
`still_pool_flashlight` (W1, W4), `flooding_corridor` (W1, W2) and
`wade_by_firelight` (W3, and the script's shots 5.5 and 6.2).

**One thing only seen on screen**: how fine the foam image must be before
its tile origin stops showing. The plan starts at four cells per tile.

### 4.4 What this does not do

Waves that rise and break, water falling between decks, and boiling or
freezing are simulation questions, not part of this. Erik's idea for a second
video (a half-sunken ship on ocean waves) would need an ocean surface outside
the ship, which is a different technique from everything here.

## 5. 3D walls and the tilt (#209, G6)

**What**: every wall tile gets a 3D piece chosen from its neighbours, for any
generated level, lit by the same light as units and props; a destroyed wall
becomes a broken piece.

Erik expects this to be fairly easy, and the kit and the neighbour rule are.
The parts that need thought are what the walls do to the picture:

- Seen from straight above, a 3D wall is only its top. Walls gain their look
  with the camera tilt, so the two are designed in one document.
- Tilted, a wall hides what stands just behind it. The design must choose:
  low walls, cut the near walls away, or fade them near the player's units.
- Where the painted level art already shows walls, the 3D pieces replace that
  part of the picture; generated ships have no painted walls, so they come
  first.
- Wall faces become surfaces for marks (section 3) and for light: a lamp on a
  wall, a shadow down a wall.

**Order inside the track**: design doc; the kit of pieces (procedural first,
so the rule can be tested before any art is made); the neighbour rule and the
destruction seam; the tilt; then the material table per ship zone.

**Judged by**: every shot in the look-shot set, before and after, plus a new
`corridor_tilted` shot.

## 6. Capture (#208) and the video-only looks (#212)

Capture: a recorded match is replayed with a director camera (a list of
keyframes) and written out frame by frame at the video's size, each frame
taking as long as it needs. Staging precise moves (walk here, look there) is
either recorded play or a scripted controller; the design decides.

Video-only looks, each a short note and a patch: the monitor feed (a second
small view from a fixed camera, with scan lines and a label), the charge's
lit timer, the blue flash, the main lights as a switch, the torn hull, the
title.

## 7. Lights

Parked by Erik's ruling until walls and tilt exist; then Claude decides
whether roof lamps and more shadow casters earn their cost. The coloured
bounce (#213) is the one light item inside this milestone.

## 8. What Claude needs from Erik

- A free machine for the first look at each track. Nothing can be judged
  without rendering.
- A look at the before and after shots at the end of each track, when he
  wants to.
- Taste calls when they come up, one at a time.

## 9. Systems

**(a) Existing systems this plan must use**

- One tone-map: every new layer writes linear HDR; nothing tone-maps itself.
- Look shots: every patch renders the named shot set before and after; new
  shots are rows in `tools/look_shots.toml`.
- Lit-3D seam (`renderer/lit3d.py`): walls and 3D marks are lit through it.
- Wall-destruction render seam (`GameMap.on_tile_changed`): how a destroyed
  wall reaches the wall kit and the marks.
- Frame lights (`renderer/frame_lights.py`): the bounce lights are entries in
  the one assembly.
- Light accessor: marks and water read the light through it.
- Smoke cast product: the water reads the smoke's shadow from it.
- Ping-pong helper (`renderer/core.py::PingPong`): the foam image.
- Particle primitive: splashes and spray.
- Static props: magazines, body parts, the wall kit's draw path if it fits.
- Dequantize convention: flow and depth are converted once at the render
  read.
- No BLAS in the frame path.
- Match recording and replay, and the camera mode value: capture.
- Match draws: any randomness in a mark is a keyed draw.
- Planted charge: the timer is a picture of its state.
- Sound layer: the exporter's cue list comes from it.

**(b) New systems, each with a draft rule**

| System | Draft rule |
|---|---|
| Marks layer | THE memory of what a match left on floor and walls; everything that leaves a mark calls its one `paint` door; render-only, never read by the simulation |
| Water flow detail | THE fine water surface: one ripple texture carried by the simulated flow; a new surface detail is a term here, never a second wave source |
| Foam field | THE foam: an amount that is added, carried by the flow and fades; a new cause of foam is a source row, never white mixed in the shader |
| Wall kit | A wall's 3D piece is chosen from its neighbours by ONE rule; a new wall look is a material row, a new shape a kit piece |
| Director camera | A film camera is a keyframe list played over a replay, as a camera mode value |
| Frame exporter | THE way a replay becomes numbered frames and a cue list; never a screen recording |
| Monitor feed | A second view of the world from a fixed camera into a texture; the only picture-in-picture path |
| Bounce lights | A lit coloured surface gives light back only as entries in the frame-light assembly |

## 10. Sources

- Alex Vlachos, *Water Flow in Portal 2*, SIGGRAPH 2010 Advances in Real-Time
  Rendering course.
  https://www.advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH%202010%20Advanced%20RealTime%20Rendering%20Course).pdf
- Nuttapong Chentanez and Matthias Müller, *Real-time Simulation of Large
  Bodies of Water with Small Scale Details*, SCA 2010.
  https://matthias-research.github.io/pages/publications/hfFluid.pdf
- Nils Thürey et al., *Real-time Simulations of Bubbles and Foam within a
  Shallow Water Framework*, SCA 2007.
  https://diglib.eg.org/items/8108fc3c-3020-4727-946f-5e5809be35fd/full
- Jesús Ojeda and Antonio Susín, *Real-time Rendering of Enhanced Shallow
  Water Fluid Simulations*.
  https://web.mat.upc.edu/toni.susin/files/hybrid_lbmsw_rend_new.pdf
- Jean-Philippe Grenier, river editor: water simulation in real time (80.lv).
  https://80.lv/articles/river-editor-water-simulation-in-real-time/
- Ian Parberry, sea foam by halftoning.
  https://ianparberry.com/research/seafoam
- Evan Wallace, *Rendering Realtime Caustics in WebGL*.
  https://medium.com/@evanwallace/rendering-realtime-caustics-in-webgl-2a99a29a0b2c
- Unity HDRP, foam in the water system.
  https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@17.0/manual/water-foam-in-the-water-system.html
- Earlier research in this repo: `docs/water_rendering_research.md`.

The papers are read from their summaries so far. Each is read in full and
archived under `docs/papers/` by the patch that implements it, and that
patch's files carry the citation.
