---
paths:
  - "renderer/unit_model_renderer.py"
  - "renderer/unit_looks.py"
  - "renderer/marine_shader.py"
  - "renderer/unit_shadow.py"
  - "renderer/held_props.py"
  - "renderer/pose_*.py"
  - "renderer/static_props.py"
  - "renderer/propgen.py"
---

# Render units: 3D units, what they hold, their pose; props

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## 3D units
- No model or animation state ever lands on `Unit` (digest!). Which model a unit is drawn with is DATA (`[render.unit_looks]`), assigned by `unit_looks.LookAssigner` the first time the renderer sees a unit and KEPT (a unit that turns zombie keeps its model). All looks share the space marine's scale, the one lit shader and no team tint; the 2D sprites are only the fallback. A new look is a table entry, never a second renderer.
- A look's colour texture carries GLOSS in its ALPHA, never coverage: the model is drawn opaque on every path, so an albedo without the channel reads as gloss 1 everywhere (#33). Each look binds the normal map baked into its own .glb (green negated: glTF's v runs down).
- Shading settings live in `[render.unit_shading]`, read every frame through `unit_looks.unit_shading`, refused when missing or mistyped; 0 / false switches each effect off.
- The projected shadow (`renderer/unit_shadow.py`) flattens the skinned model along the light at ONE elevation for every light, at one fixed far depth under `GL_LESS`, so each floor pixel darkens once and no shadow covers a body; it reads the light only through the accessor's `LightView`; floor plane only.
- A unit's BODY is lit from its one per-unit light read (`unit_shadow.read_unit_light`), never from the tile under each fragment (#114); the smoke cast shadow stays per fragment.
- HANDEDNESS is a look trait, render-only (#187, Erik's ruling: it must not affect play), dealt by `unit_looks.HandAssigner` and kept for life, never on `Unit`. A left-hander is the mirror image of the pose, its mesh and textures unmirrored. A handedness that matters in play is a separate gameplay ruling.

## Held props
- A held prop is a row of `[render.held_props]` (`weapons` maps the loadout weapon id to a `props.<name>` row), every key through the refusing door. Drawn ONLY with the pose layer on, at body @ hand bone @ grip, through the units' ONE lit shader and in the unit's projected shadow: never a second light GLSL, never baked into a character. A look whose skeleton lacks the socket holds nothing.
- WHICH socket is `PoseLayer.socket_bone`: a hand switch is that row, never a renderer branch.
- A left-handed unit holds the right-hander's prop on the partner bone at the DERIVED grip `held_props.mirrored_grip`: a proper transform, never a reflected (inside-out) one, and never a configured left-hand grip.

## Pose layer
- THE only place a 3D unit's bone pose is composed: clips, layers, crossfades, the additive layer and post-FK modifiers all blend through it in ONE batched `compose` per look per `draw_units` call (FK as 4x4 products per depth level, never a per-bone Python loop); a `Skinner` takes its output to the mesh. Never a second pose path.
- Skeleton-agnostic: no bone name is ever a literal in `renderer/pose_clips.py`, `renderer/pose_layer.py`, `renderer/pose_modifiers.py`, `renderer/pose_posture.py`, `renderer/pose_locomotion.py` or `renderer/held_props.py` (tested). A state is a `[render.pose.states]` row, playback a `[render.pose.clips]` row; a new skeleton is a table (its clip names, rate, clips, roles, states, hit directions, sockets, locomotion, mirror pairs), never code.
- Locomotion: every looping clip's stride speed is MEASURED at load (`pose_locomotion.measure_gait`, THE one measurement); a gait clip plays at the unit's drawn speed over its stride speed, NEVER a literal gait rate; the gait is chosen on the clips' rates, never a speed in m/s (`tests/test_pose_locomotion.py`: no skating). A hit's direction is the event's own `dir_x_q` / `dir_y_q`, never a render-side guess (#185).
- MIRROR (`pose_clips.MirrorRig`): the mirror image is `G'_j = M G_{m(j)} A_j`, exact for ANY rest pose; never a bare `M G M` swap (SOMA's rest pose is asymmetric). A mirrored clip is a `mirror_of` row baked once at load; a left-handed unit is composed in the mirrored world; a right-handed row is untouched bit for bit (`tests/test_pose_mirror.py`). A new mirrored thing is a `mirror_of` row or a direction reflected at the layer's door, never a second mirror path.
- Reactions are data: a new reaction or modifier is a row or a `PoseModifier`, never a branch in the renderer. POSTURE is a profile row scaled by the unit's amount `a`; per-unit variation draws from a LOCAL generator seeded by `(seed, unit id)`, never the sim's or numpy's global; `a = 0` composes bit-identically to no posture. A new carriage is a profile row, never a code path.
- Per-unit state lives on `_UnitAnimState.pose`, never on `Unit`; sim events queue per unit and are acted on once (the re-delivered `tick_events` list is guarded). An additive's delta is against its clip's own `additive_ref` key, never the bind rest.
- The reader refuses CUBICSPLINE, matrix nodes and more than one skin by name; the raylib 5.x / 6.x `ModelAnimation` split goes through `pose_skinner.anim_pose_fields` (`tests/test_pose_inject_gl.py`, on each machine).

## Static props
- No prop render state ever lands on a sim entity (digest!). Sway = tamed wind × the tile's gas DENSITY (momentum flux, not velocity: Erik's vented-room ruling); calm or evacuated air means motionless props.
- Prop assets live in `assets/models/props/` with a license file per pack; OBJ preferred (raylib 5.5's cgltf rejects 2020-era GLBs).
