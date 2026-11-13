# Tracers — design (2026-11-12, #141)

Erik: the bullets "show up as quite ugly lines"; he asked for "tracers that
move and glow, and perhaps sparks where the bullets hit".

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | A round is a white-hot STREAK with a halo and an afterglow, in the linear HDR scene, coloured through the blackbody ramp | The old line was drawn after the tone-map and could not glow; this matches the muzzle flame |
| 2 | HONEST speed by default (`speed_scale = 1`), Erik's choice: "I'd like to try the honest speed actually, I might change my mind" | A rifle round (900 m/s) crosses a room inside one or two frames; the streak is the path flown during one exposure (`shutter_s`). `speed_scale` below 1 slows it for the eye, at the price of the picture lagging the sim |
| 3 | The speed is the round's own (`[ammo.*] muzzle_mps` through the weapon's ammo family), never a render dial | "Honest" must stay honest when a round's row changes |
| 4 | Sparks and a pinpoint flash where a round stops on a WALL, fired when the streak arrives | Cause and effect; with honest speed that is the first frame |
| 5 | What a round stopped on is read render-side from `solid` at the path's end | No sim file changes, so no determinism or Class-P gate moves |
| 6 | No new shader: a tracer is up to four sprites of the particle draw | The Particle primitive row; the capsule sprite is already a streak |
| 7 | A round's first segment starts at the shooter's muzzle | Flame and tracer leave the same point |
| 8 | The legacy flat line is deleted | One picture of a round |

## As played (2026-11-12)

Erik tried honest speed and landed on `speed_scale = 0.20` and `halo_gain =
0.54` ("ok, I'm happy"). The speed is still derived from the round's own
`muzzle_mps`; the scale is the one look dial on it.

## Not in this patch

A round that hits a unit or a cover crate throws nothing (a body hit deserves
its own look); the laser beam is still the legacy line; a tracer casts no
room light.

## Gates

`tests/test_tracers.py` (door, honest kinematics, gap-free coverage, the
muzzle start, one per round, the cap, wall impacts once on arrival, the pack's
layout and its light when zoomed out). HUMAN-TEST: Erik plays it before merge.

## Systems

**(a) Used**: Particle primitive (the sprite draw, `MuzzleSparkSystem`),
Blackbody GPU LUT, Muzzle flame (the muzzle pose, the sim clock), One
tone-map. **(b) New** — the Tracers row in CLAUDE.md.
