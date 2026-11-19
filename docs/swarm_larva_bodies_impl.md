# Swarm larva bodies: as built (arc #63, 2026-11-19)

Erik's order (2026-11-19): "In code sounds good. Yes pls go ahead". The
larvae were drawn only as debug dots; they now have bodies, made in code.
Render-only: nothing in `src/` or `cpp/` changed.

## What was built

- **The grub** `renderer/propgen.py::generate_larva` (+ `larva_profile`):
  the top half of a segmented tube along +x, `rings` interior rings
  (Chebyshev-spaced, so the ends round off), a half-ellipse of four points
  each (three strips): `6 x rings` triangles, 36 at the shipped 6. A
  heightfield from above (no triangle hides another, so no depth test),
  every triangle wound counter-clockwise from +y. Blunt tail, pointed head
  (`tail_taper`, `head_taper`). The propgen contract (verts, normals,
  colours u8) with RGB a baked belly shade and alpha the along-body
  coordinate (data). Not in `GENERATORS` (that registry is the placed
  props'). **Home**: propgen is THE procedural geometry source and its rule
  says new procedural geometry is a generator there, never inline mesh
  code elsewhere; a larva is a creature, not a prop, but the reason for the
  rule (one place for procedural geometry, deterministic, no `simulation`
  import) holds for it.
- **The view** `renderer/larva_bodies.py`: the `[render.larvae]` door
  (every key required, ranges the shader relies on refused by name), the
  pure packers (`pack_mesh`, `pack_live`, `DeadLarvae`, `instance_texels`),
  the render state `LarvaBodies` and the GL face `LarvaBodyDraw`.
- **The draw**: ONE `glDrawArraysInstanced` over the empty VAO
  (`GLDirect.DrawArraysInstanced`, new, plus `Uniform3f`). The template's
  vertices come from a `(n, 2)` RGBA32F mesh texture (`gl_VertexID`), each
  larva's two float4s from a 1024-wide instance texture (`gl_InstanceID`,
  larva i at column `i % 1024`, rows `2 (i // 1024)` and `+1`).
- **The wriggle** (`shaders/larva_body.vs`): one travelling wave per larva,
  `a = 2 pi waves s - phase`; peristalsis moves points along the body by
  `P L sin(a) / (2 pi waves)` (stretch `1 + P cos a`, cross-section
  `/ sqrt(stretch)`: a squeezed segment bulges), a small sway moves them
  sideways; normals follow. The phase is `2 pi frac(walked metres /
  stride_m + hash(unit_id))` from the store's `dist_walked`: a crawler
  undulates at its own speed, a still larva holds its pose. No clock.
- **The light** (`shaders/larva_body.fs`): THE lit-3D block spliced at the
  `// <COMMON>` / `// <FIELD_SAMPLE>` markers (light field, smoke cast
  shadow, light curve), half-Lambert, a Blinn gloss and a rim of the local
  light, segments as shaded grooves (darker, normal tilted along the body),
  a darker head capsule, a cold tint in chill coma. Linear HDR,
  premultiplied, drawn in the dot view's slot (after the water, before the
  floor smoke).
- **Death**: from `SwarmUnitDiedEvent` (pre-zero position and heading), the
  body goes limp over `limp_s` (wriggle out, flattened to `dead_flatten`,
  spread, darkened to `dead_rgb`, gloss gone), then fades over `fade_s` on
  the sim clock; it starts from the phase it had alive (kept from the last
  live pack by unit id), so the pose does not jump. Capped at `dead_cap`.
- **Wiring** (`renderer/game_renderer.py`): `_upload_larva_bodies` in
  `upload_state` (door read every frame, view-culled pack), deaths in the
  event consumer, `_draw_larva_bodies` at compose slot 2b, unload.
- **Config**: `[render.larvae]` (28 dials, `enabled`); `[debug] swarm_dots`
  now `false`.
- **Look shots**: `larva_nursery`, `larva_closeup`, `larva_studio_wide`.
- **Rules**: `.claude/rules/render-units.md` "Larva bodies" (+ paths),
  the dot view's line in `.claude/rules/swarm.md`, one index line and the
  region row in `CLAUDE.md`.

## Decisions

- **Dots default OFF** (`[debug] swarm_dots = false`); when on they draw
  over the bodies (a debug picture). No reason found to keep them on.
- **Death flecks**: not drawn by the bodies view. The limp, darkening,
  fading body is the death's mark; the red-brown fleck stays the dot
  view's alone (it is still drawn when the dots are on).
- **Size**: `length_m = 0.10` (0.30 of a tile, the dot view's reach),
  `radius_m = 0.021`. The species row's comment says ~6 cm; 6 cm reads as
  a grain at the world target's 48 px a tile (`~9 px`). Erik may want the
  honest size.
- **Per-fragment light** (an accepted gap, written in the rule): a 3D
  unit's body reads ONE per-unit light (#114); a larva is a third of a tile
  long and reads the tile under each fragment, as the debris does.
- **No depth test**: the grub is a heightfield; overlapping larvae stack in
  instance order, the dead under the living.

## Numbers

- Triangles: 36 a larva (rings 6). Mesh texture 108 x 2 texels.
- Frame cost, `tools/e2e_drive.py --profile --glfinish`, RTX 3070, 1280x720
  hidden window, 200 frames (vsync bound, 16.7 ms median throughout):

  | scene | before (dots) upload + draw | after (bodies) upload + draw |
  |---|---|---|
  | larva_studio (600 larvae) | 0.28 + 0.19 ms | 0.43 + 0.84 ms |
  | 2 000 larvae (a temporary copy of the studio, two broods of 1 000) | 0.35 + 0.21 ms | 0.46 + 0.85 ms |

  The cost is flat in the count (CPU per draw: uniforms and the lit-3D
  seam's setters; ~0.28 ms in isolation, about 0.17 ms more is
  `GameRenderer._light_field_ctx`, which the debris also calls). The GPU
  part of 2 000 x 36 triangles did not show above the noise. Mean frame
  17.1 to 17.8 ms in every run.

## Tests and deliberate breaks

`tests/test_larva_bodies.py` (pure, 20) and `tests/test_gl_larva_bodies.py`
-> `tests/gl_larva_bodies_check.py` (6 GL checks in a subprocess). Each
break below was applied, its test went red, and it was reverted:

| break | red |
|---|---|
| an extra `glDrawArrays` beside the instanced draw | GL `one_draw_whatever_the_count` |
| the phase reads `time.perf_counter()` | `test_a_still_larva_keeps_its_phase...` |
| an emissive term in the fragment shader | GL `lit_by_the_light_field` |
| the packer writes `dist_walked` | `test_the_view_reads_only_its_declared_columns...` |
| the upload reads `gmap.tile_size_m` before the off test | `test_off_reads_nothing_and_packs_nothing` |
| the shader's instance row stride 1 instead of 2 | GL `one_draw...` (larva 2999 of 3000 not drawn) |
| the dead pruned at twice their life | `test_the_instances_are_exactly_the_living_and_the_fading_dead` |
| straight alpha instead of premultiplied | GL `a_faded_body_leaves_the_floor_untouched` (first form, on an unlit limp body, stayed GREEN: rewritten to a lit pale body, off by 0.41 then) |
| the shader ignores the phase | GL `the_pose_is_a_function_of_the_phase` |
| the bodies drawn after the dots removed (draw call deleted) | `test_compose_world_draws_the_bodies_on_the_floor_under_the_smoke` |
| a re-typed field-sample block | `test_the_shader_is_lit_through_the_lit3d_block...` |

## Findings (not fixed)

- `tools/look_shots.toml`'s `dark_corridor_lamp` is not reproducible on
  this machine: two renders of the same commit (b1c51f23, no change)
  differ in ~5 900 pixels (rows 136 to 827, x from 481). `fire_plume` and
  `blast_final_t84_frag` are bit-identical before and after this patch.
- `GameRenderer._light_field_ctx` costs ~0.17 ms a call and is built once
  per consumer per frame (the debris, now the larvae); one per frame would
  do.
- The `lit3d` uniform setters call `GetShaderLocation` every frame (five
  per larva draw); caching them in the seam would help every consumer.

## Left out / not verified

- No C++ build, no CUDA gate, no full suite (the machine was busy with
  another gate run): run on a borrowed build of the same sources
  (`63-to-main/cpp/build/Release`, b1c51f23). `config.toml` changed, so
  the full suite is the gate before a merge.
- The wriggle was judged only in stills and by the GL checks; Erik judges
  the motion in `larva_studio` (`main.py --level larva_studio --control
  direct`).
- No death was seen in a game shot (nothing kills larvae in the studio's
  first seconds without a fire on fuel); the dead look is in the GL
  close-up only.
- No turning bend (a larva that turns snaps its heading as the law does),
  no eating pose, no shadow cast by a larva, no per-species look table
  (one species exists).
