# Octopus control layout (for a per-limb controller)

Prototype, 2026-10-27. Render-only: these are float pose controls, not
simulation state, and not deterministic gameplay data.

## The call

```python
import octopus_math as om
points, rotations = om.pose_octopus(body: om.BodyParams, arms: list[om.ArmParams] * 8, lod="game")
```

Pure Python + NumPy, deterministic, no Blender, no game. It returns one head
position and one rotation (relative to the bind pose) per bone, in
`om.bones(lod)` order (`"game"`: 142 bones, `"hero"`: 270). A value outside its
range raises `ValueError`; nothing is clamped or extrapolated.
`pose_octopus(..., with_scale=True)` also returns one scale per bone: the
thinning of a stretched arm. A skin that drops it shows a stretched arm at its
rest thickness.

## Per arm: seven floats (`ArmParams`), arms in order L1 L2 L3 L4 R1 R2 R3 R4

L = the animal's left. 1 is the front pair and 4 the back pair. Every sign is
mirrored, so the same numbers on L_n and R_n give a mirror-symmetric pose.

| control | range | what it does |
|---|---|---|
| `curl` | 0 .. 1 | Total bend along the arm, growing toward the tip (the proof's curl law, 5.35 rad at 1). 0 is a straight arm. |
| `bend_dir` | -1 .. 1 | Which way the arm bends, as an angle around its own axis in units of pi. 0 bends toward the suckers (oral side), +0.5 toward the animal's front, -0.5 toward its back, and ±1 bends dorsally (upward). It is periodic: -1 and +1 are the same pose. |
| `twist` | -1 .. 1 | Axial twist spread evenly along the arm, ±100° at the tip. Together with `curl` it turns a planar curl into a 3-D helix and rolls the suckers. |
| `tip_curl` | -1 .. 1 | Extra bend on the distal 45 % of the arm, in the same plane as `curl`. A negative value makes an S-curve or flicks the tip. |
| `lift` | -1 .. 1 | Root pitch, ±40°. A positive value raises the arm off the ground. The pitch is bent in smoothly over the first 6 joints (the root), not pivoted at one point. |
| `swing` | -1 .. 1 | Root yaw inside the crown, ±18° at the root (the tip moves ±14° around the crown). A positive value swings the arm toward the BACK, on both sides (measured 2026-10-27; this table said "front" before). It is bent in over the same root joints. |
| `stretch` | 0 .. 1 | Makes the arm longer and thinner, as a real octopus arm does when it reaches. 0 is the rest length. 1 is the size's `stretch_max`: the free arm 1.4 times as long, the whole arm 1.31 times (2.52 m to 3.31 m for the medium size). The arm keeps its volume: its thickness shrinks as 1 / √(length factor), so at full stretch it is about 85 % as thick. Only the free arm stretches. The part inside the crown and the junction never move, and the stretch fades in over the first three segments past the junction. |

All seven respond smoothly. Each of the 32 segments keeps exactly the length
the size and `stretch` give it, so bending never stretches an arm. Reaching
is pointing the arm (`lift`, `swing`, `curl` → 0) plus `stretch`.
`stretch` does not change what the other six mean: `curl` 1 is still 5.35 rad
of total bend, so a stretched arm curls into a wider coil.

**Independence.** Arm k's bones depend only on `arms[k]` and on the two shared
body controls `crown_spread` and `body_pitch`. `verify_octopus.py` checks this
by moving one arm and confirming the other seven stay bit-identical. For
`stretch` it also checks the skin itself: changing arm k's stretch moves only
arm k's own free-arm vertices. Every other arm, the crown, the web, the
junction rings, the eyes and the siphon stay bit-identical.

## Body: six floats (`BodyParams`)

| control | range | what it does |
|---|---|---|
| `mantle_squeeze` | 0 .. 1 | Jet contraction: the side and top mantle walls move in by up to 6.5 cm. |
| `mantle_pitch` | -1 .. 1 | Tilts the mantle about the neck by ±20°. A positive value raises it. |
| `siphon_yaw` | -1 .. 1 | Aims the siphon ±70°. A positive value aims it toward the animal's left. |
| `siphon_pitch` | -1 .. 1 | Aims the siphon ±45°. A positive value aims it up. |
| `crown_spread` | -1 .. 1 | Shared root pitch for all arms, added to each arm's `lift`. -1 closes the crown 80° down into a bundle (jet swimming), +1 opens it 25° up. The response is smooth through 0. |
| `body_pitch` | -1 .. 1 | Pitches the whole animal ±90° about the crown. At -1 the oral face points forward and the mantle leads. |

## Size and proportions (the SIZES table)

How big the octopus is, and how long its arms are against its body, are not
fixed numbers. Each size is one row of a small table, `SIZES` in
`scripts/octopus_math.py`. The row in use is named by `SIZE`. Switching is one
value: `om.set_size("big")`, or the environment variable
`OCTOPUS_SIZE=big` for a build. A baby or a boss will be one more row.

A row has three dials:

| dial | what it means |
|---|---|
| `body_length_m` | How long the body is in metres, from the front of the head to the back of the mantle. Everything else scales with it. |
| `arm_body_ratio` | How long an arm is, from its root inside the crown to its tip, divided by the body length. The arm's thickness stays the body's, because the arm grows out of the crown. So a higher ratio gives longer, more slender arms, and a lower ratio gives shorter, stouter ones. |
| `stretch_max` | How much longer the free part of an arm (past the crown) gets at `stretch` = 1. |

Erik's ruling of 2026-10-28: every size keeps option A's proportion, **arms
1.8 times the body length**, and the same stretch. So the three sizes are the
same animal at three scales. There is no fixed span any more: the span follows
from the body length.

`stretch_max` is 1.40 for every size. Only the free arm stretches (the part
inside the crown does not), so the whole arm, root to tip, gets 1.31 times
longer: the medium octopus's 2.52 m arm reaches 3.31 m. At full stretch an arm
is 0.85 times as thick.

| row | what it is | body length | mantle (wide x long) | arm at rest | arm stretched | arm thickness at the root |
|---|---|---:|---:|---:|---:|---:|
| `small_preview` | option A's body, at the family proportion | 1.00 m | 0.66 x 0.78 m | 1.80 m | 2.36 m | 0.21 m |
| `mid` | **the default, the medium size** (option B's body) | 1.40 m | 0.93 x 1.08 m | 2.52 m | 3.31 m | 0.30 m |
| `big` | a preview of a later larger size, "the big baddie" (option C's body, as long as a marine is tall) | 1.885 m | 1.25 x 1.46 m | 3.39 m | 4.45 m | 0.40 m |

Tip to tip, with all eight arms straight (a marine is the game's space marine,
1.885 m; a game tile is 0.333 m):

| row | span at rest | span fully stretched |
|---|---|---|
| `small_preview` | 3.82 m = 2.0 marines = 11.5 tiles | 4.93 m = 2.6 marines = 14.8 tiles |
| `mid` | 5.34 m = 2.8 marines = 16.0 tiles | 6.91 m = 3.7 marines = 20.7 tiles |
| `big` | 7.19 m = 3.8 marines = 21.6 tiles | 9.30 m = 4.9 marines = 27.9 tiles |

(The renderer draws the marine 6 tiles tall, a little more than his true
1.885 m; on that ruler the spans are 6 % more tiles.) In the rest pose the
arms are curled, so the medium octopus covers about 5.1 x 3.4 m of floor.

`proof` is the animal before #106 (authored units, the 1.2 m arm) and is kept
so it can be rebuilt. The three #106 options (`mid_bulk`, the 1.24-ratio
`mid`, `mid_heavy`) are in git at commit 65df56a. Pictures:
`previews/bodyplan/`.

## What a controller does not get (yet)

- **No contact, collision or ground.** Arms can pass through the floor, the body and each other. Both suckers and contact are visual only.
- **Some full-range combinations fold the crown skin.** An example is every arm at `lift` +1 together with `crown_spread` +1, which is 65° up. The arm tubes stay sound across the whole range. The crown and web between the arm roots is what folds: in the random full-range checks the worst pose had 72 inverted slivers out of 12k triangles. The shown poses fold at most 8. Treat |`lift` + spread| > 45° as "looks bad up close".
- **Shading of posed arms in the game is not exact.** raylib's CPU skin gets
  the surface normals of every posed limb wrong (measured, all looks, not just
  the octopus; worse on a stretched arm). It did not show at the playground's
  light level, but a brightly lit close-up may. See the README.
- **No squeezing.** Nothing makes the body narrower than it is. The game's
  human doors are 3 tiles (1.0 m) wide in most levels and 4 tiles (1.33 m) in
  the playground; a few side doors are 2 tiles (0.67 m). The medium octopus's
  mantle is 0.93 m wide and its crown, where the arms join, 1.10 m. So it fits
  a 4-tile door as it is, a 3-tile door only with its mantle (not its crown)
  and a 2-tile door not at all. The big one (mantle 1.25 m) fits none of them.
  Squeezing is a later piece of work.
- **No speed limits.** A controller that jumps a control from one value to another in a single frame makes the pose jump too. Rate limiting belongs in the controller.

## Example poses

`om.named_poses()` holds rest, curled, reach (L1 alone, stretched to 0.8), swim and walk.
`om.swim_cycle(t)` and `om.walk_cycle(t)` are demonstration loops, not
behaviour.
