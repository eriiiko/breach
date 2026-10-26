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

## Per arm: six floats (`ArmParams`), arms in order L1 L2 L3 L4 R1 R2 R3 R4

L = the animal's left. 1 is the front pair and 4 the back pair. Every sign is
mirrored, so the same numbers on L_n and R_n give a mirror-symmetric pose.

| control | range | what it does |
|---|---|---|
| `curl` | 0 .. 1 | Total bend along the arm, growing toward the tip (the proof's curl law, 5.35 rad at 1). 0 is a straight arm. |
| `bend_dir` | -1 .. 1 | Which way the arm bends, as an angle around its own axis in units of pi. 0 bends toward the suckers (oral side), +0.5 toward the animal's front, -0.5 toward its back, and ±1 bends dorsally (upward). It is periodic: -1 and +1 are the same pose. |
| `twist` | -1 .. 1 | Axial twist spread evenly along the arm, ±100° at the tip. Together with `curl` it turns a planar curl into a 3-D helix and rolls the suckers. |
| `tip_curl` | -1 .. 1 | Extra bend on the distal 45 % of the arm, in the same plane as `curl`. A negative value makes an S-curve or flicks the tip. |
| `lift` | -1 .. 1 | Root pitch, ±40°. A positive value raises the arm off the ground. The pitch is bent in smoothly over the first 6 joints (the root), not pivoted at one point. |
| `swing` | -1 .. 1 | Root yaw inside the crown, ±18°. A positive value swings the arm toward the front. It is bent in over the same root joints. |

All six respond smoothly and keep the arm's length. Each of the 32 segments
keeps its length exactly, so "reach" means pointing the arm (`lift`, `swing`,
`curl` → 0), never stretching it. Real octopus arms do stretch; that is not
modelled.

**Independence.** Arm k's bones depend only on `arms[k]` and on the two shared
body controls `crown_spread` and `body_pitch`. `verify_octopus.py` checks this
by moving one arm and confirming the other seven stay bit-identical.

## Body: six floats (`BodyParams`)

| control | range | what it does |
|---|---|---|
| `mantle_squeeze` | 0 .. 1 | Jet contraction: the side and top mantle walls move in by up to 6.5 cm. |
| `mantle_pitch` | -1 .. 1 | Tilts the mantle about the neck by ±20°. A positive value raises it. |
| `siphon_yaw` | -1 .. 1 | Aims the siphon ±70°. A positive value aims it toward the animal's left. |
| `siphon_pitch` | -1 .. 1 | Aims the siphon ±45°. A positive value aims it up. |
| `crown_spread` | -1 .. 1 | Shared root pitch for all arms, added to each arm's `lift`. -1 closes the crown 80° down into a bundle (jet swimming), +1 opens it 25° up. The response is smooth through 0. |
| `body_pitch` | -1 .. 1 | Pitches the whole animal ±90° about the crown. At -1 the oral face points forward and the mantle leads. |

## What a controller does not get (yet)

- **No contact, collision or ground.** Arms can pass through the floor, the body and each other. Both suckers and contact are visual only.
- **Some full-range combinations fold the crown skin.** An example is every arm at `lift` +1 together with `crown_spread` +1, which is 65° up. The arm tubes stay sound across the whole range. The crown and web between the arm roots is what folds: in the random full-range checks the worst pose had 72 inverted slivers out of 12k triangles. The shown poses fold at most 8. Treat |`lift` + spread| > 45° as "looks bad up close".
- **No speed limits.** A controller that jumps a control from one value to another in a single frame makes the pose jump too. Rate limiting belongs in the controller.

## Example poses

`om.named_poses()` holds rest, curled, reach (L1 alone), swim and walk.
`om.swim_cycle(t)` and `om.walk_cycle(t)` are demonstration loops, not
behaviour.
