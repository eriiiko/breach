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
| `stretch` | 0 .. 1 | Makes the arm longer and thinner, as a real octopus arm does when it reaches. 0 is the rest length. 1 is the size's `stretch_max` (about 1.4 times as long for the mid size). The arm keeps its volume: its thickness shrinks as 1 / √(length factor), so at full stretch it is about 85 % as thick. Only the free arm stretches. The part inside the crown and the junction never move, and the stretch fades in over the first three segments past the junction. |

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
value: `om.set_size("mid_heavy")`, or the environment variable
`OCTOPUS_SIZE=mid_heavy` for a build. A baby or a boss will be one more row.

A row has three dials:

| dial | what it means |
|---|---|
| `body_length_m` | How long the body is in metres, from the front of the head to the back of the mantle. Everything else scales with it. |
| `arm_body_ratio` | How long an arm is, from its root inside the crown to its tip, divided by the body length. The arm's thickness stays the body's, because the arm grows out of the crown. So a higher ratio gives longer, more slender arms, and a lower ratio gives shorter, stouter ones. |
| `stretch_max` | How much longer an arm gets at `stretch` = 1. |

The three "mid" rows are the three readings of "the body is about a marine's
size". Each spans 2.0 marines from arm tip to arm tip with the arms straight at
rest, and 2.5 marines with them fully stretched (a marine is the game's space
marine, 1.885 m):

| row | option | body length | mantle (wide x long) | arm at rest | arm thickness at the root | arm / body | stretch_max |
|---|---|---:|---:|---:|---:|---:|---:|
| `mid_bulk` | A: body like a marine in bulk, long arms | 1.00 m | 0.66 x 0.78 m | 1.78 m | 0.21 m | 1.78 | 1.34 |
| `mid` | **B: the default**, in between | 1.40 m | 0.93 x 1.08 m | 1.73 m | 0.30 m | 1.24 | 1.40 |
| `mid_heavy` | C: body as long as a marine is tall, short heavy arms | 1.885 m | 1.25 x 1.46 m | 1.68 m | 0.40 m | 0.89 | 1.50 |

All three spans are 3.77 m at rest and 4.71 m stretched. In the game the
marine is drawn 6 tiles tall, so that is 12 tiles at rest and 15 tiles
stretched. At a strict 0.333 m per tile it would be 11.3 and 14.2 tiles.

The arm is about one marine long in every option. What changes is how big the
body is, and so how thick the arms are. `proof` is the animal before this
change (authored units, the 1.2 m arm) and is kept so it can be rebuilt.
Pictures: `previews/bodyplan/`.

## What a controller does not get (yet)

- **No contact, collision or ground.** Arms can pass through the floor, the body and each other. Both suckers and contact are visual only.
- **Some full-range combinations fold the crown skin.** An example is every arm at `lift` +1 together with `crown_spread` +1, which is 65° up. The arm tubes stay sound across the whole range. The crown and web between the arm roots is what folds: in the random full-range checks the worst pose had 72 inverted slivers out of 12k triangles. The shown poses fold at most 8. Treat |`lift` + spread| > 45° as "looks bad up close".
- **Shading of posed arms in the game is not exact.** raylib's CPU skin gets
  the surface normals of every posed limb wrong (measured, all looks, not just
  the octopus; worse on a stretched arm). It did not show at the playground's
  light level, but a brightly lit close-up may. See the README.
- **No speed limits.** A controller that jumps a control from one value to another in a single frame makes the pose jump too. Rate limiting belongs in the controller.

## Example poses

`om.named_poses()` holds rest, curled, reach (L1 alone, stretched to 0.8), swim and walk.
`om.swim_cycle(t)` and `om.walk_cycle(t)` are demonstration loops, not
behaviour.
