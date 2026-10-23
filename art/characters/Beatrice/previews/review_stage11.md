# Beatrice, round B2b (head ×1.20, face re-trace, toon look): independent review

2026-10-20. I judged by eye from the pictures in `previews/`. I took the B2a set from git (`0d9d90f`) for
before/after. I read no builder notes. The face references are `concept-face-front-v1.png` and
`concept-face-side-v1.png`. For style, figure and how the latex is painted I used `concept.jfif` and
`with_hair*.png`. Crops and zooms are in `C:/tmp/beatrice_rev11/v/`. The ones I name are `cmp_front.jpg` /
`cmp_prof.jpg` (drawing | model with hair | model without hair, cut from `face_vs_art.jpg`), `ft0/30/60.jpg`
(face_turn tiles at 2×), `fc_eyes_toon.jpg` / `fc_eyes_art.jpg` / `fc_3q.jpg` / `z_eyecorners.jpg` (from
`face_closeup.jpg`), `zeb30.jpg` (face_zebra 30° at 1.6×), `z_prof_ear.jpg`, `pvc_torso.jpg`, `boxes.jpg` and
`ft0_chest.jpg`. The hair is judged only for how it frames the face. The body's shape did not change and is
not re-judged.

Measurements: on `face_vs_art.jpg` the drawing's tile and the model's tile share one frame (420 × 380 px a
tile), so I measured features in tile pixels on 2× gridded copies. I measured hair and head widths as runs
of pixels that differ from the background (summed RGB difference > 45). I took colours as 7 × 7 means. Suit
tone is luminance percentiles in fixed boxes on the bust and belly of `posed_vs_concept.jpg`, with the grey
background masked out of the render.

---

## 1. The face against the close-ups

### Proportions: right

The geometry now matches the front drawing. Each pair below is drawing / model, in tile px.

| | Drawing | Model |
|---|---|---|
| Eye centres (spacing) | 161 and 272 (111) | 163 and 272 (109) |
| Eye opening, width × height | about 65 × 30 | about 62 × 30 |
| Nose-tip highlight | y 207 | y 200 |
| Nostrils | y 218 | y 221 |
| Mouth line | y 249, 40 wide | y 248, 37 wide |
| Chin point | y 298 | y 296 |
| Face width at y 200 | about 190 | about 195 |

The profile outline (`face_vs_art.jpg`, right column, red line) lies on the drawing's brow, nose, lips and chin
to within a line's width. **What still separates the face from the drawing is drawing and paint, not form.**
The one measured exception is the iris. It is about 10% narrower than the drawing's (33 vs 37 px).

### Eyes: the biggest gap left in the face

Pictures: `face_closeup.jpg` top row, `fc_eyes_toon.jpg` against `fc_eyes_art.jpg`, `z_eyecorners.jpg`.

- **Shape and angle: good.** The new shape is near-level, rounder and has a short flick. It reads as the same
  eye as the drawing's. The B2a eye was a long tilted almond, and that is gone.
- **No lashes.** The upper lid is one flat, even black ribbon with a separate fish-tail flick. A small step
  shows where they meet (`z_eyecorners.jpg`, left eye, top left). The drawing's upper lid is a thick
  **brown lash mass**, feathered, with single lashes fanning out at the outer third. Above it runs a thin
  **double-lid crease**. Along the lower lid it has a row of short lower lashes. The model's lower lid is a
  faint brown dotted line that disappears at normal distance (`face_turn.jpg` 0°).
- **No lid shadow.** In the drawing, the upper lid throws a soft dark band across the top third of the iris
  and the top of the white. That band is what makes her look intent and slightly hooded. In the model the
  iris is evenly lit right up to the liner, and the white is plain white (B2b made it whiter). Together with
  the missing lashes, the eyes read as a **wide, blank doll stare**. The drawing's gaze is serious. B2a's pbr
  eyes did have this grey under the lid (`face_closeup.jpg` bottom row).
- **Iris:** a little small (above). It is a lighter lavender-blue and lacks the drawing's dark outer ring and
  darker top. The pupil is small. The two catch-lights are fine.
- **Corners.** At the outer end, the white stops in a blunt vertical edge with small stair-steps instead of
  tucking under the lid. On her left eye a tiny "^" notch sits at the lower outer corner. At the inner end,
  the liner hooks steeply down to the lower lid. The drawing's inner corner is a fine taper with a touch of
  pink. All of this shows in `z_eyecorners.jpg`. In three-quarter view the outer corner of the white looks
  torn (`fc_3q.jpg`, right of the iris).

### Brows

They are in the right place: y 128–131 against the drawing's 125–135. They have the wrong weight and angle.
The model's brows are thin, grey, hatched pencil lines, level and slightly long. The drawing's brows are short,
thick and dark brown, with the inner end lower and heavier. That gives the art its slightly knitted, serious
expression. With thin level brows over open white eyes, the model looks mildly surprised. This is cheap to
fix and changes the expression more than anything else of its size.

### Nose

The B2a tunnel and dark diamond are gone, and the profile nose fits the drawing. From the front, though, the
nose is now just a **small round highlight dot and two nostril dots** (`cmp_front.jpg`, `fc_eyes_toon.jpg`).
The drawing paints a large, pale, **teardrop-shaped** highlight about 1.5× wider, with a sharp white glint
inside it and two curved nostril-wing strokes (`face_closeup.jpg`, "drawing: nose and mouth"). The
three-quarter nose line (`fc_3q.jpg`) works well.

### Mouth

The model's mouth is a short thin line with a faint pink smear above it. The drawing has a clearly tinted
**lower lip** with a soft gloss highlight, a darker upper-lip line and slightly wider corners. In profile the
drawing's upper lip sits a little forward, the lower lip tucks back, and there is a small dip under it. The
model's profile runs from nose to chin as one smooth curve with a barely visible notch (`face_closeup.jpg`,
"mouth, from her right" against "drawing: profile").

### Cheeks, jaw, chin

The outline matches the drawing. The jaw and chin read right from the front. In three-quarter view the chin
looks a little long and pointed (`ft30.jpg`), mostly because nothing (hair, shadow) covers the jaw corners as
it does in the drawing.

### Skin shading

- **Ramp and terminator colour: right.** The lit skin is (242, 204, 186) against the drawing's
  (230, 190, 174), slightly lighter. The one shadow step is warm, (221, 168, 152), the right hue. The drawing's
  deepest skin shadow, under the chin, is (191, 137, 121). The model never goes that deep, so the under-chin and
  neck separate from the face mainly by the outline.
- **Blush: about 40% of the drawing's.** Measured as the drop in green from base skin to blush, it is 22 in the
  drawing and 9 in the model. The drawing's blush is also wider and runs right under the lower lashes.
- **The fringe's shadow** shows as soft, blurry, vertical grey streaks running down from the fringe tips between
  and over the eyes (`fc_eyes_toon.jpg` centre, `fc_3q.jpg`). They read as smudges or creases in the skin, not as
  a shadow. The drawing puts a crisper darker band on the forehead and upper lids that hugs the fringe's tips.
  For a toon look the shadow should be a clean shape one ramp step down, not a blur.
- **Outlines: good.** The face, jaw, ear and neck lines are thin warm-brown lines of about the drawing's
  weight, and the three-quarter nose line works. Keep them.

## 2. Three-quarter views: the cheek "dent"

It is real geometry, not a shading artefact. In `face_zebra.jpg` at 30° (`zeb30.jpg`) the reflection lines
bunch and kink along a groove. The groove runs from below the outer corner of the eye, down and back, to the
corner of the jaw. In the toon look it shows as a **light crescent next to a darker band** on that cheek, at
30° and 60° (`ft30.jpg`, `ft60.jpg`) and in `head.jpg` at default framing, right of the eye. It reads as a
hollow cheek or a fold.

In these pictures it is on the cheek **nearer the camera**, the broad one on the ear side. The narrow cheek
beyond the nose is clean. The front zebra also shows small closed islands on both cheeks under the eyes, so the
surface is lumpy there in general. Smooth the cheek from the eye to the jaw corner, or carry the "simple head"
shading normals all the way back to the ear. It is small, but it shows at the default head framing.

## 3. The toon look on the suit and boots against `concept.jfif`

**At full figure it works** (`hero.jpg`, `turnaround.png`, `posed_full.jpg`, `torso.jpg`). Crisp white anime
highlights on a near-black base read immediately as wet latex. The boots read as patent, the same material as
the suit. The knee pads are no longer grey mirror portholes. They are now dark ovals with a pale rim. They still
look flat, a sticker rather than the drawing's dome with a curved streak (`knee.jpg`), but they no longer pull
the eye.

**What the concept paints that the toon look does not** (`pvc_torso.jpg`, `boxes.jpg`): a **broad soft sheen**.
The concept's bust is a large mid-grey gradient that turns smoothly into dark at the underbust, with one small
hot spot on each breast. Luminance in the bust box:

| Percentile | 25th | 50th | 75th | 90th | 95th | 99th |
|---|---|---|---|---|---|---|
| Concept | 30 | 40 | 96 | 172 | 200 | 241 |
| Render | 26 | 29 | 54 | 61 | 65 | 232 |

The render is two flat steps (about 29 and 60) plus white flecks. It has nothing between 65 and 230, which is
exactly where the concept's whole sheen lies. The form of the bust and belly therefore comes only from the
outlines and the flecks. The concept's base is also a neutral-to-warm charcoal, mid-tone (38, 34, 38). The
render's is a touch cool, (28, 27, 32).

**The upper-chest blobs** (`ft0_chest.jpg`, `head.jpg`, `head_size.jpg`, `face_vs_art.jpg` row 1): confirmed.
Below the collar on each side of the zip sit large pale blue-white patches with ragged, torn edges, each about
the size of the mesh inserts beside them. They read as spilled paint or holes cut in the suit, not as gloss.
They are the hard specular threshold catching the broad, nearly flat upper chest. In the concept that area
is the *darkest* part of the torso, under a thin soft sheen. A related item: the long white stripe down her
left flank, from ribs to thigh (`hero.jpg`, `torso.jpg`), is so even and continuous that it reads as a printed
racing stripe. The concept's rims are thinner and broken.

## 4. The whole figure in the concept's framing

`posed_vs_concept.jpg`, `posed_hotlight.jpg`. She is clearly closer than after B2a. The face is now the same kind
of drawing as the concept's, and the suit reads as anime latex. In the concept framing her eye-to-chin height is
about 0.32 of her shoulder width, against 0.27 in the concept. Part of that is the concept's lowered chin. The
rest is that her hair shows the whole long face. What dominates the remaining gap, in order:

1. **Hair silhouette and framing.** The concept's hair is a wide, layered mass, wider than the head from the
   cheeks down and flaring onto the shoulders. The model's hair is a narrow helmet with straight curtains. At the
   eye line the model's hair is 245 px wide against the drawing's 356 (69%). At the chin it is 242 against 388
   (62%), and it does not widen at all toward the shoulders (`face_vs_art.jpg` front tiles). The narrow helmet,
   the comb-tooth fringe and the exposed ears make the head read small and the face long.
2. **Head pose and gaze.** The concept's head is tilted to her right and the chin is lowered, so she looks up
   from under her brows. The render stands upright and looks straight ahead. Half of the concept's intensity is
   this pose. It costs nothing in modelling.
3. **The eyes' weight** (section 1): no lash mass and no lid shadow.
4. **The latex's broad sheen** (section 3).
5. Design differences that are the owner's call, not judged here: the concept's high-cut bare hip, and the front
   waist inserts (review 10, item 4).

## 5. Is the anime look an improvement over pbr?

**Yes, clearly, for the whole character.** Look at `look_compare.jpg` and `face_turn.jpg` against B2a's. The face
was plastic and pale, with a shaded wedge nose. It is now flat warm skin that belongs to the same drawing as the
eyes. The hair reads as anime hair rather than black rubber tubes. At full figure the suit's crisp highlights
read as wetter latex than pbr's dim satin, which was review 10's item 3.

What pbr did better, and the toon look should take over:
- **The broad graded sheen on the suit** (`posed_hotlight.jpg`, both pbr panels). The bust and shoulders carried
  a large soft grey gradient, which is the concept's value structure. Add it as a soft gradient or a third ramp
  step for sheen under the hard white highlight, and limit the white to small, curved, high-curvature spots.
  That also removes the chest blobs.
- **Shadow under the upper lid** on the white of the eye and the top of the iris (`face_closeup.jpg` bottom row).
- **A soft rim of light** along the arms' outer edges. The concept has this too.

Nothing else from pbr is worth keeping. Its face shading, the hair, and the suit reflecting the head were all
worse.

## 6. What the hair round must achieve (priority order)

1. **Front fringe.** Replace the evenly spaced comb of equal triangular spikes, which stops at the brows with
   skin between every tooth. Use **5–7 clumped, tapered locks of varying length**, dense enough that the forehead
   shows only in slivers. The drawing parts the fringe slightly to her right of centre. One **central lock drops
   between the eyes to the lash line or nose bridge**, and the outer locks reach the outer eye corners. The
   fringe must overlap the upper lids and brows, and cast a crisp toon shadow on them. This alone changes the
   face most.
2. **Side locks.** From the temples, long locks should fall **in front of the ears**, close to the face, down over
   the cheeks and past the jaw corners. They should cross the face outline below the cheekbone, as both drawings
   show, and hide most of each ear. Today the side curtains hang straight from above the ears, well clear of the
   face. Both ears stick out, and a flat pointed tab stands in front of each ear (`face_turn.jpg` 60–135°,
   `face_vs_art.jpg` profile).
3. **Crown and volume.** Height on top, with a visible part and strands sweeping from it. The hair mass should
   widen from cheek level down to the shoulders (target: about 1.4× today's width at the eye line, 1.6× at the
   chin, per section 4). The highlight band should be a narrow, bright "angel ring" with sawtooth edges that
   follow the strands. Today it is a broad, dull, wavy grey smear across the front of the crown (`head.jpg`).
4. **Colour range.** Lit hair in the drawings goes warm ash-brown and much lighter. The model's hair is a flat
   dark brown.
5. **Back.** Layered locks with separated, tapered, slightly wavy ends, falling to about mid-back, wider at the
   shoulders (`with_hair_back.png`). Today it is one flat sheet down to the waist, with a visible step at the
   nape where the cap meets the hanging hair (`face_turn.jpg` 180°).

## 7. Ranked corrections for the next rounds

1. **HAIR: fringe and side locks to frame the face** (section 6, items 1–2). **Worth a round** (it is the planned
   hair round).
2. **FACE: the eyes' drawing.** Upper lash mass with a fanned outer third, double-lid crease, lower lashes, a soft
   lid shadow over the top of the iris and the white, a slightly greyer white, the iris about 10% larger with a
   dark outer ring and a darker top, and outer corners tucked under the lids instead of the stair-stepped edge.
   `fc_eyes_toon.jpg` against `fc_eyes_art.jpg`, `z_eyecorners.jpg`. **Worth a round** (it could share one with
   item 3).
3. **FACE: brows thicker, darker and shorter, inner end lower.** `face_closeup.jpg` top row. **Polish**, but the
   cheapest expression win.
4. **LOOK: latex sheen.** Add a broad soft sheen step to the suit's ramp and limit the hard white to small spots.
   That removes the ragged upper-chest blobs and breaks up the flank "racing stripe". Warm the base slightly.
   `ft0_chest.jpg`, `boxes.jpg`, `posed_hotlight.jpg`. **Worth a round** (a lookdev pass).
5. **FACE: the cheek groove** from the outer eye to the jaw corner on the near cheek in three-quarter view, plus
   the lumpy islands under the eyes. `zeb30.jpg`, `ft30.jpg`, `head.jpg`. **Polish**, but do it before showing.
6. **FACE: paint.** Nose highlight as a larger teardrop with a glint and nostril-wing strokes. The lower lip
   tinted, with a gloss highlight, and a slight pout in profile. Blush about 2× stronger and wider.
   `face_closeup.jpg`. **Polish.**
7. **LOOK: shadows.** A crisp, one-step fringe shadow instead of the blurry streaks, and a deeper under-chin and
   neck shadow, toward the drawing's (191, 137, 121). `fc_eyes_toon.jpg`, `ft0.jpg`. **Polish** (do it with the
   hair round, since the fringe changes).
8. **OTHER: pose the head in the concept framing.** Tilt to her right, lower the chin, eyes up to camera.
   `posed_vs_concept.jpg`. **Polish** (posing only).
9. **OTHER: the hairless pictures.** In `face_vs_art.jpg` row 2 profile (`z_prof_ear.jpg`), the hidden hair seems
   to still cast its shadow: a ragged terracotta band runs from the crown behind the ear down to the jaw. A pale
   outlined **ghost ear** also sits behind the real ear. Check whether the ghost ear is geometry or a picture
   artefact, and drop the hidden hair's shadows from those pictures. As it stands, the hairless views cannot be
   used to judge the skull's skin. **Polish.**
10. **LOOK: knee pads** read as flat dark stickers with a pale rim. The drawing has a dome with a curved streak
    (review 10, item 2, still open in shape). **Polish.**

## 8. Verdict

**Is the face ready to show the owner as "close"? Not yet. Close in structure, not yet in drawing.**
B2b did the hard part. The proportions match the front close-up to within 2–3 px and the profile outline fits.
The eye shape is right, the head size is right, and the toon look puts every part of her in one anime style.
That is a large step from B2a. But the owner's bar is "as good as in the picture", and what he looks at in a
face is the eyes, the brows and the hair framing. All three are still unlike the drawing.

**What he will notice first:**
- **The hair helmet** with its comb-tooth fringe and the two exposed, sticking-out ears (`face_vs_art.jpg`,
  `head.jpg`). It makes the face long and the head small.
- **The eyes stare.** There are no lashes and no lid shadow, so the eyes read blank and doll-like next to the
  drawing's heavy-lashed, serious eyes. The thin level brows add to it.
- In `head.jpg` and `face_turn.jpg` 0°: **the white paint blobs on her upper chest**, right under the face.

Showing her after the hair round, with items 2, 3 and 5 folded in, would be the right moment to call it close.
Showing her now is fine only as a check of the toon direction. On that he should say yes.
