# Beatrice, round B1i (hands, wrist cuffs, knee pad and boot-shaft polish) — independent review

2026-10-08. I judged by eye from the pictures in `previews/`, with the B1h set taken from git (`73ab55b`) for
before/after. I read no builder notes. I opened every picture named below whole, and cropped, gridded or
contrast-stretched the places in question (crops in `C:/tmp/beatrice_rev9/v/`). References:
`without_hair.png` (the hand reference), `without_hair_side.png`, `without_hair_back.png`. The low heel and
the placeholder head are not judged.

Measurements: `sheet.png`, `side.png` and `back.png` share the drawings' 1536-px frame (1 px is about
1.2 mm). For the hands I measured on `sheet.png` against `without_hair.png`, row by row, both hands, in two
ways: a skin mask (red minus blue > 25), and everything that differs from the background. The drawing's ink
outline adds about 1 px a side to its widths. Positions I read by eye are from 4x crops with a 10-px ruler
(`sheet_hand_L.jpg`, `sheet_hand_R.jpg`) and are good to about ±3 px. The side ortho panels of
`hands_before_after.jpg` (the drawings' scale ×4) gave a second length check. The leg widths use review 8's
method (outermost dark pixels per row), so they compare with its table directly.

---

## 1. The stage-7 and stage-8 items

### B2. Hands and cuff (review 7)

| Item | Now | Where it shows |
|---|---|---|
| **Surface ring-banded** | **Fixed.** No rings on the back of the hand or the fingers, in matte or in gloss. In the zebra view the back of the hand carries broad, smooth reflection lines and the fingers carry clean lengthwise lines (B1h: stacked closed loops). One flaw is left: a band of tight, wavy lines runs straight across the back of the hand where the fingers start. That is a crease or ledge (see section 2). | `hands_before_after.jpg` "reflection lines", "matte"; `zebra.jpg`; `hj_root.jpg` |
| **Finger share of the hand: 41% vs 52% drawn** | **Fixed.** Front view, from the knuckle corner on the outer edge of the hand to the fingertip: drawing 741→808 of 680→808, **52%**. B1i 755→828 of 697→828, **56%**. Both hands are curled the same way in this view, so the shares compare. B1h's palm-box-with-stubs look is gone. If anything the fingers are a touch long now. | `sheet_hand_L.jpg`, `sheet_hand_R.jpg` |
| **Hand length right** | **Kept.** From the top skin row under the cuff to the lowest fingertip: drawing 128 / 127 px (her right / left), B1h 132 / 131, B1i 131 / 130 (+2%). In the side ortho the hand is longer: 124 px against 114 drawn, cuff to tip (+9%). Most of that is the drawing's fingers curling away from the viewer there while B1i's hang almost straight. The whole hand still hangs about 17–20 px (2 cm) lower than drawn, because the cuff does (B7, unchanged). | `meas.py` output; `side_panels.jpg` |
| **Wrist slightly thick (+11%)** | **Fixed, and now slightly thin.** The narrowest point under the cuff, by the background method: drawing 30 px (about 28 without its outline), B1h 27, B1i **25**. The palm then widens to 43 px (drawing 42). The wrist is now about 3 mm slimmer than drawn while the cuff above it is the drawn 35–37 px, so the cuff is 1.4× the wrist (drawing 1.15×). That is part of why the cuff reads as a bracelet (see B2 cuff below). | `meas` tables; `cuff_cmp.jpg` |
| **No thumb reads** | **Fixed.** The thumb hangs in front of the curled fingers, with its nail showing, in `sheet.png` (both hands), `wrist.jpg`, `torso.jpg` and `hero.jpg`. Its tip is 85 px below the cuff (drawing 90). | `sheet_hand_L/R.jpg`; `wrist.jpg` |
| **Pose: clumped, little finger splayed, mannequin** | **Partly.** The clump is gone. The fingers now cascade and curl toward the thumb, as asked. But they curl too far and spread too wide. In the front views (`sheet.png`, `torso.jpg`, `hero.jpg`, `legs.jpg`) the middle, ring and little fingers point sideways at the thigh, almost horizontal, with a finger's width of air between each pair. Seen from three-quarter front they show as separate short stubs that seem to come out of the palm (`t_rh5.jpg`), which reads as a claw or a rake. In the drawing the fingers hang **down**: the curl tips them inward perhaps 20–30°, and they lie close together and overlap like a fan. | `torso_hands_cmp.jpg`, `hero_hands_cmp.jpg`, `t_rh5.jpg`; `sheet_hand_L/R.jpg` vs the drawing |
| **Cuff: hard double-ringed tube, dark gap on the little-finger side** | **Partly.** The double ring and the gap are gone. It is one band, seated, and the skin goes in under it with no dark slot (`cuff_cmp.jpg`, B1h vs B1i). Its height (about 21 px against the drawn 20) and its step up from the sleeve match the drawing. What is left is that it still reads as a **stiff ring**: straight vertical sides, a flat, wide underside that shows as a broad brownish rim from below (`hand.jpg`, `hj_cuff2.jpg`), and it stands well off the now slimmer wrist (`wrist.jpg`, `wr_cuff.jpg`). The drawing's cuff is a glossy turned-back band whose lower edge lies close on the wrist. | `cuff_cmp.jpg`; `hj_cuff2.jpg`; `wr_cuff.jpg` |
| Good (length, placement, skin shading, cuff height) | **Kept.** The skin shading is clean and even, with no blotches, even contrast-stretched. | — |

### Review 8, corrections 1–2 (knee pad, boot shaft)

| Item | Now | Where it shows |
|---|---|---|
| **1a. Pad too dark (pad/suit luminance 1.0 vs 1.33 drawn)** | **Fixed by the number, not the way the drawing does it.** Mean luminance in `sheet.png`, pad box against thigh box: drawing 53 / 42 = **1.28**, B1h 0.98, B1i 39 / 31 = **1.29**. But the drawing's pad has the *same* base tone as the suit (median 41 against 41) and is lighter because of a bright central highlight and a bright rim (95th percentile 135). B1i lifted the base tone (median 41 against the suit's 31) and has almost no highlight (95th percentile 52). So the pad reads as a flat, satin, grey disc: in `legs.jpg`, `knee.jpg` and `hero.jpg` it looks like smoked glass or a grey sticker, not a glossy shield. The rim is there and shows as a thin light line. | `knee_front_cmp.jpg`; `legs.jpg`; `knee.jpg`; `legs_polish_before_after.jpg` front |
| **1b. Dome 18 mm proud, 7–10 mm ahead of the knee line, 17-px step below** | **Fixed in substance.** Side depth across the knee: 116 px (B1h 121, drawing about 111). The knee front now lies on the drawing's line: 439 / 446 / 456 px at rows 940 / 970 / 990 against the drawing's 439 / 448 / 458 (after the 19-px offset). The step back onto the shin is now 11 px over 20 rows (B1h 16, drawing 6). In `knee_side_cmp.jpg` the outline is one convex knee line with a soft shoulder at the pad's lower edge. The shell look is gone. | `knee_side_cmp.jpg`; `legw.py side` |
| **2a. Ankle bones (front 49 → about 55 px at 1320–1350)** | **Mostly fixed.** Front width at 1340 / 1350: 52 / 52 px (B1h 47 / 47, drawing 56 / 53). The swell is on both sides, centred at 1340–1350, and gone by 1370. In the back view the shaft swells from 46 to 51 px at 1290–1310. In the zebra views the reflection lines now bend around the ankle instead of running straight down. In `boot.jpg` the left boot's highlight is an S, not a stripe. In `boot_side.jpg` the outer bone shows as a soft dome with two highlight spots close under the shaft top. It reads as an ankle bone, a little high and round, not as a lump. | `legs_polish_before_after.jpg` "reflection lines"; `boot.jpg`; `boot_side.jpg`; `legw.py front/back` |
| **2b. Hollow above the heel cup** | **Partly.** It shows in the reflections (`legs_polish_before_after.jpg` back zebra; `boot_side.jpg`) but not in the outline. The side silhouette's back edge is the same column, x = 553–555 from row 1280 to 1360, in B1h and B1i. The heel cup therefore still rises out of a straight back line in profile. | `side.png` rows 1250–1370 |

---

## 2. The hands as a whole, at close-up quality

They are much better: banding gone, proportions right, thumb present, skin clean. But they do not yet read as
a slim woman's relaxed hands at close-up. What makes them read as doll-like, in order of visibility:

1. **The pose from the front and three-quarter front** (`torso.jpg`, `hero.jpg`, `legs.jpg`, `torso_back.jpg`):
   the fingers splay out sideways like a claw (section 1, "Pose"). Both hands do it. It is the first thing on
   the hands at full-body distance, and it is worse than B1h's clump because it adds a gesture the drawing
   does not have.
2. **The finger roots are a ledge.** The back of the hand ends in a straight line across, with a dark notch
   between each finger. The fingers look plugged onto the end of a paddle, like a mitten edge with glove
   fingers (`hand.jpg` at y≈520; `hj_root.jpg`; the zebra band across the knuckles). The drawing's back of the
   hand runs into the fingers over the knuckle heads, with the webbing lower than the knuckles.
3. **The fingers are tubes.** They have the same width from root to tip, no knuckle bumps, no joint creases
   and a blunt rounded end (`hj_tips.jpg`, `side_panels.jpg`). The drawing's fingers taper to the tip and show
   their joints. In matte (`hands_before_after.jpg` "matte" B1i) this reads as rubber.
4. **The back of the hand is flat and wide**, with no metacarpal fan or tendons (`hand.jpg`, `wrist.jpg`). It
   is the drawn width (43 vs 42 px), but because it is uniform and flat it looks puffy next to the slim wrist.
5. **Nails are painted-on decals** (`nails.jpg`): flat pink rectangles with a straight base and no cuticle
   curve, flush with the skin. On the ring finger in `hand.jpg` the nail shows end-on as a mauve-grey disc
   that looks like a bruise. The thumb's free edge is a grey band. In `hand_palm.jpg` a pink rectangle floats
   on a curled finger. The drawing's nails are small, domed, pale and almond-shaped, set in the finger.

Left and right are mirror images and show the same faults. The palm view (`hand_palm.jpg`) is acceptable: a
smooth palm, curled fingers, the thumb hanging separately. Its fingertips are the same blunt tubes.

## 3. The cuffs

Seated and clean: no gap, no double ring, no wavy skin edge (`cuff_cmp.jpg`). The height is right. It still
reads as a **stiff tube**: parallel straight sides, a flat bottom face that shows as a wide rim from below
(`hj_cuff2.jpg`), and a cuff/wrist ratio of 1.4 against the drawing's 1.15. Making the lower edge roll in onto
the wrist (a rounded lip, no flat face) and giving back a millimetre or two of wrist would make it a band.
The step at its top edge is drawn too and is fine.

## 4. Knee pads and boot shaft after the polish

- **Pads:** no longer dark eggs, and no longer a shell in profile. The new problem is the finish: a light,
  flat grey disc with a thin bright rim. At `legs.jpg` and `hero.jpg` distance they read as **smoked-glass
  ovals**, and they are now the lightest dark-grey shapes below the hips, so they draw the eye about as much as
  before, for a different reason. The drawing gets its lighter pad from gloss: suit-dark base, a strong
  vertical highlight down the middle and a bright piped rim. Taking the base tone back toward the suit's and
  making the pad as reflective as the boots would match it.
- **Ankle:** it reads as an ankle, not as lumps. The reflections curve round it. Only the back profile is still
  a straight line above the heel cup.

## 5. Did anything change outside the hands, cuffs, knee pads and boots?

No. Old against new, pixel difference > 12 (PNG) / > 20 (JPEG):
- `sheet.png`, `side.png`, `back.png`: changes only in rows 650–849 (hands and cuffs), 900–1049 (knee pads) and
  1250–1399 (boots), plus 3 px at a sole.
- `flank.jpg`, `shoulder.jpg`, `shoulder_back.jpg`, `armpit_back.jpg`, `head.jpg`: identical (max 6–10 grey levels,
  JPEG noise).
- `hero.jpg`, `torso.jpg`, `legs.jpg`, `torso_back.jpg`, `hero_back.jpg`, `side.jpg`, `top.jpg`, `elevated.jpg`:
  every changed band lies on the hands, cuffs, knees or boots (checked in 50-row bands).
- `flank_back.jpg`: changes only at its bottom-left, where the hand is.

Nothing was broken.

## 6. Verdict

**Legs: yes, close to the pelvis's quality now.** The knee is one line in profile and the boot has an ankle.
What remains is a finish choice on the pad (grey satin instead of glossy dark) and a straight back profile
above the heel. Both are small.

**Hands: not yet.** The round fixed what review 7 measured: banding, finger share, thumb, wrist, the cuff gap.
The hands now have the right proportions, but at close-up they read as a mannequin's: tube fingers plugged
onto a flat paddle, decal nails, and from the front a splayed, claw-like curl. The pelvis came out of its rounds
with nothing reading as an add-on. Here the finger roots and the nails still do, and the cuff is a stiff ring.

**What the owner will notice first:** the hands' pose in `torso.jpg` and `hero.jpg`. The fingers stick out
sideways like a claw, where the drawing's hang down, close together. Second, in `legs.jpg`/`hero.jpg`, the pads
as grey smoked-glass ovals. Third, at close-up (`hand.jpg`), the fingers as tubes with painted nails.

## 7. The next round: upper body surface

I re-checked review 7's B3, B4, B6 and B7 in the current pictures. All are unchanged, as intended (the
pictures are pixel-identical above the hands). One finding from `sleeve_ripples_cause.jpg` changes the plan:
the ripples are confined to the **shoulder patch** (a separate surface from the deltoid down to mid-arm). The
sleeve below it is a clean loft, and the patch's lower boundary *is* the mid-arm "garter" ring of B7. B4 and
B7 are one job: rebuild the shoulder/upper-arm as one clean surface and put the seams where the drawing has
them.

Ranked by how much each spoils a hero or close-up shot:

1. **Shoulder and upper-arm gloss ripples (B4).** A hammered, orange-peel mottle on the deltoid highlight and a
   ladder of horizontal streaks down the back of the upper arm (`shoulder.jpg` right deltoid, `sh_delt.jpg`;
   `flank_back.jpg` at 2x, `fb_tip.jpg`; `armpit_back.jpg` upper arm). The drawing has smooth, sweeping
   highlights over a rounded deltoid. Cause: the patch's field, not its mesh (`sleeve_ripples_cause.jpg`).
   Remake the patch the way the sleeve below it is made.
2. **Mid-arm ring → the drawn seams (B7, same job as 1).** A horizontal raised ring with a V dip at mid upper
   arm (`shoulder.jpg`, `flank.jpg`, `armpit_back.jpg`, `torso_back.jpg`). The drawing (`without_hair_back.png`,
   `arm_cmp.jpg`) has a diagonal raglan seam over the shoulder, high outside and low at the armpit, and an
   elbow seam with a soft fold. Add about 5% to the upper arm and a rounder deltoid while there.
3. **Collar (B3).** A straight cylinder with a ledge at its base that casts a crisp line on the chest, and a dark
   inner lip at its top (`shoulder.jpg`, `sh_collar.jpg`). The surface crumples where the shoulder panel's seam
   meets the collar base on the viewer's right. The drawing's collar hugs the neck at the top and flares into
   the trapezius with no edge at its base.
4. **Back-armpit tears (B6).** Where the shoulder-blade insert ends at the arm, the seam is jagged and crumpled,
   with a small nub sticking out (`armpit_back.jpg` 470–530 × 220–360; `torso_back.jpg` both armpits). It
   reads as a tear. The drawing has a clean seam curving into the armpit.
5. **Doubled lines and sawtooth piping (B6).** On the chest inserts, the insert's own piping and a separate seam
   run side by side with a gap and end in a zigzag (`shoulder.jpg`, `sh_insert.jpg`). At the upper tip of the
   shoulder-blade insert the piping zigzags (`fb_tip.jpg`). The drawing has one clean piped outline per insert.
6. **Mesh material (B6).** Grey with coarse light dots about 5 mm apart: a perforated-metal sticker
   (`sh_insert.jpg`, `flank.jpg`). The drawing's mesh is a fine dark net with warm skin tone showing through
   (`without_hair.png` chest and waist inserts).
7. **[hand leftover] Finger pose and finger form.** Carry this in, or better, give it its own short round
   before the upper body, since it is what the owner will see first. Hang the fingers down and close together,
   curled only about 20–30° inward and overlapping like the drawing's fan (`torso_hands_cmp.jpg` vs
   `without_hair.png`). Blend the finger roots over knuckle heads instead of a ledge (`hj_root.jpg`). Taper the
   fingers and give them joints (`hj_tips.jpg`). Make the nails small domed almond plates set into the
   fingertip, with no grey tip and no mauve disc (`nails.jpg`). Roll the cuff's lower edge onto the wrist, and
   give the wrist 1–2 mm back (cuff/wrist 1.4 → about 1.15; `hj_cuff2.jpg`).
8. **[leg leftover] Knee-pad finish.** Suit-dark base, glossy, a strong central highlight and a bright rim,
   instead of a grey satin disc (`knee_front_cmp.jpg`, `legs.jpg`). Optionally the hollow above the heel cup
   in the side outline (`side.png` 1280–1360). A few minutes each.

Keep the hips and the bust out until the owner has ruled. The boot piping pattern (review 8 item 3) is still
his design question.
