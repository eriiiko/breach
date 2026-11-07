---
name: light-curve
description: Tuning Breach's light curve with Erik at the screen -- the Photoshop-style Curves map on the light (how bright, how contrasty, how cartoony the dark and the lit parts look) and its live on-screen editor. Covers opening the panel, reading it, the recipes (threshold, darker dark, grey dark), what it cannot do, and how to land a tuned curve. Load before changing `[render.light_curve]`, or when Erik wants to change how the darkness or the lit areas look.
---

# Tuning the light curve

*Status: PROVISIONAL. Written 2026-11-07 from G5 (#64), where Erik tuned the first shipped curve by
hand. Every run amends it before merging (see "Keeping this skill honest").*

The curve is a map from the light reaching a pixel to the light that pixel takes instead. It runs on
the floor, the units and the props. It scales the light, so a red lamp stays red, and it acts before
the art and the normal map, so their detail survives. It is render-only: no sim state, digest or
golden moves, so tuning it needs no physics session.

Code: `renderer/light_curve.py` (spec and numpy twin), `lit3d._LIGHT_CURVE_GLSL` (the one GLSL copy),
`ui/light_curve_panel.py` (the editor's decisions), `ui.draw.draw_light_curve_panel`.

## Open the panel

1. In `config.toml`, `[render.light_curve]`, set `panel = true`.
2. Start the game on a level with the lights to judge, or press Ctrl+R if it is running. The panel
   is top right. `fire_smoke_studio` shows fire, smoke and marines together:

       <breach-py> main.py --level fire_smoke_studio --windowed

Erik needs these commands exact, with this machine's python (`docs/dev_setup.md`). Run from a
worktree, the game needs `cpp/build/Release/*.pyd` copied from the main checkout first.

## Read it

- **Left to right:** the light a spot gets from the game. **Bottom to top:** how bright it is drawn.
- **On the diagonal** nothing changes. Above it brighter, below it darker. Steep is contrast, flat
  is everything the same brightness.
- **Grey bars:** how much of the room sits at each light level. A change where there are no bars
  touches nothing, so the same curve can do little in one room and a lot in another.
- The readout over the plot gives the light under the mouse and what it becomes.

Drag a dot, click empty space to add one, right-click to delete. The first and last dot stay.
**on/off** compares with the plain light, **reset** gives the straight line and full colour, **save**
writes the curve back. **dark colour** greys the art in the dark: 1.0 is off, 0.15 the first
prototype's grey.

## Recipes

- **Threshold, the cartoon look:** a dot on the diagonal, a second just right of it dragged near the
  top. Soft light stays below the jump, light above it goes fully lit. Two dots with very different
  slopes on each side make a corner, so the line stays straight right up to the jump.
- **Darker dark:** pull the left part below the diagonal.
- **Room lift:** raise the left or the middle. A tile with exactly no light stays at the flat floor
  whatever the curve does; that floor is `[render.lighting] floor`.
- **Brightness plateau:** flatten the right part, so everything bright reads alike. Light above
  `range` carries on rising with the top of the curve's ratio, so a very bright source still shows.

## What it cannot do

- **It sees brightness, not the source.** Light arrives as one summed field, so a lamp as bright as
  the flashlight snaps with it. Only a threshold above the lamps singles the flashlight out.
- **A steep jump exposes the 16 directions.** A lamp's soft glow turns into a hard octagon (seen
  2026-11-07, `dark_corridor_lamp`). Tile steps and the fire's flicker can show the same way.
- **There is no bounce.** Light outside a cone is the sweep's spread, smoke glow and the flat floor.
  Bounce is designed (ray-engine design v3 §7.4) and not built.
- **Smoke and water read the plain light,** so they do not follow the curve.
- **The editor is in the game only,** not in `lighting_demo`. Look shots do honour the saved curve.

## Landing a curve

**Save** rewrites `enabled`, `points` and `dark_saturation` in the `config.toml` of the checkout the
game runs from, and nothing else. It does not touch `panel`: set `panel = false` by hand before
committing, or the shipped game opens with the editor. Read `git diff config.toml` and commit only
the curve's lines when Erik has other tuning there.

Points are relative to `range`, so changing `range` moves every point: a tuned curve stays tuned only
at the `range` it was tuned on. The fire's `[render.fire_light] gain` and the units' look were judged
before this curve, so recheck them if one looks off.

A look change is a HUMAN-TEST: Erik plays it and says so before it merges.

## Keeping this skill honest

Every run that uses this skill amends it in its own branch before merging: what was skipped, what
failed, what was learned. One dated line is enough. A step is a MUST only with its incident beside
it.

- 2026-11-07 (G5, #64): Erik tuned the first curve in `fire_smoke_studio` and loved the tool. He first
  could not read the panel from a list of controls; what worked was the axes, then one drag.
