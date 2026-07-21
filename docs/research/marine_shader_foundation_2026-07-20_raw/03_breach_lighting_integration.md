# Breach lighting integration — foundation for a lit 3D-marine shader

Reconnaissance for making 3D marines (`renderer/unit_model_renderer.py`) consume
the SAME lights the 2D ship is lit by, so they match the scene. Read-only; no
code changed. All refs are `file:line` in the `anim-phase0` worktree.

TL;DR: Breach does NOT light the ship from raw light-source positions. The C++
raycaster bakes every source (with occlusion, colour, 1/r falloff) into a
**per-tile field** — RGB incoming colour + a 2D dominant light direction — and
the ship's fragment shader samples that field and does N·L against a normal map.
The parity-correct path for marines is to sample the **same baked field** on the
3D mesh's real normals, not to re-light from `LightSource` uniforms.

---

## 1. How is light computed?

**The field is a per-tile RGB intensity + a per-tile 2D direction, produced on
the CPU by the C++ raycaster each frame.**

`LightingPass` (`renderer/lighting.py:42`) owns the CPU scratch buffers:

- `light_rgb` — `(h, w, 3)` f32, the RGB incoming-light accumulator
  (`lighting.py:52`).
- `light_dx`, `light_dy` — `(h, w)` f32, the accumulated light-direction vector
  components (`lighting.py:53-54`).
- `light_map` — `(h, w)` f32 scalar, **derived** as `light_rgb.max(axis=2)` — a
  brightness proxy for the render-side unit/smoke tinting consumers
  (`lighting.py:57`, computed at `lighting.py:298`). This is the scalar the
  sprites and the current 3D `light_fn` read; it is NOT the field the ship
  shader uses.

`compute_light_field()` (`lighting.py:225`) zeroes the buffers, then for each
source calls the C++ raycaster `cast_source_directional(...)`
(`lighting.py:286-291`), which DDA-marches rays and accumulates intensity +
direction per tile. After the loop it normalizes the direction field to unit
vectors (`normalize_directions`, `lighting.py:294`) and repacks.

- **Where it runs:** CPU-driven — the renderer owns the cast. C++ does the
  per-ray march (`cpp/src/raycaster.cpp:91` `cast_source`, and the directional
  variant), Python owns the numpy buffers and the per-frame loop. The march is
  scheduled to move into the sim in "S5" (`lighting.py:258-259`), but today it
  is a render-side per-frame cast.
- **What a cell represents:** RGB incoming light colour (a physical 1/r power
  field, ray-count-independent — engine/08 "pure-density march",
  `lighting.py:110-115`) PLUS a unit-length 2D direction of dominant incoming
  light. Occlusion, coloured gas absorption, and distance falloff are all
  already resolved INTO the cell (`lighting.py:238-266`).
- **How often:** every frame, inside `GameRenderer.upload_state`
  (`game_renderer.py:285-312`) when `show_lighting` and `light_sources` are
  present. Otherwise the field is zeroed (`game_renderer.py:313-327`).

The field is then packed into two RGBA16F GPU textures (`lighting.py:300-317`):

- **Texture A** (`light_tex_a`): `RGB = light_rgb`, `A = light_dir.x` (signed).
- **Texture B** (`light_tex_b`): `RGB = smoke_glow` (god-ray, drawn separately),
  `A = light_dir.y` (signed).

16F stores HDR RGB and signed direction directly — no 0.5-centred encode
(`lighting.py:300-303`). These textures are physics-resolution `(w, h)`, created
at `lighting.py:65-66`, bilinear-filtered by default (`lighting.py:73`,
`toggle_bilinear` at `lighting.py:216`).

---

## 2. How is the SHIP lit? (the precedent to match)

**Per-fragment diffuse + normal-map N·L against the baked per-tile light field.
It uses the field's colour + baked 2D direction, NOT source positions.**

`LightingPass.draw_lit_world()` (`lighting.py:321`) draws the diffuse texture
over the full world RT inside `BeginShaderMode(self.shader)` where `self.shader`
is `shaders/lighting.vs` + `shaders/lighting.fs` (`lighting.py:76-79`). It binds
these samplers/uniforms each draw (`lighting.py:350-356`):

Samplers:
- `u_diffuse` — texture0, implicit (raylib convention), the sRGB ship art.
- `u_normal` — normal map at diffuse resolution (linear).
- `u_light_a` — the RGBA16F field A (RGB colour, A = dir.x).
- `u_light_b` — the RGBA16F field B (RGB smoke_glow, A = dir.y).
- `u_vacuum` — physics-res mask; vacuum tiles are `discard`ed.

Scalar uniforms (set in `LightingPass.__init__`, `lighting.py:81-119`):
- `u_ambient` (vec3) — flat light floor (`lighting.py:129`, default
  `(0.18,0.18,0.22)`; game overrides to `(0.10,0.10,0.13)` at `main.py:298`).
- `u_normal_strength`, `u_use_normal`, `u_normal_y_sign` (+1 GL / -1 DX),
  `u_srgb_decode`, `u_light_z` (0 = grazing/high relief … 1 = straight down),
  `u_light_gain` (render exposure), `u_art_uv_rect`.

The fragment shader (`shaders/lighting.fs`):

1. Computes `world_uv` from `fragTexCoord` and `u_art_uv_rect` (fs:85).
2. Discards vacuum tiles (fs:89-92).
3. Samples diffuse (art-space UV), sRGB→linear (fs:94-99).
4. **Samples the light field at `world_uv`** (fs:107-108):
   `incoming_rgb = tex_a.rgb`, `light_dir_2d = vec2(tex_a.a, tex_b.a)`
   (fs:109-111). This is **per-fragment sampling of a per-tile field** —
   bilinear-filtered, so the ship gets a smooth lighting gradient across tiles.
5. Unpacks the normal map, flips Y by `u_normal_y_sign` (fs:114-119).
6. **Builds the 3D light direction from the baked 2D dir + `u_light_z`**
   (fs:126): `L = normalize(vec3(light_dir_2d, u_light_z))`. So the "third
   dimension" of the light is a global lamp-height dial, and the in-plane
   direction comes from the raycast field.
7. `ndotl = max(dot(N, L), 0.0)` (fs:128).
8. Composite (fs:133): `lit = diffuse * (u_ambient + incoming_rgb *
   u_light_gain * ndotl)`, then ACES tone-map (fs:71-78, 136) and linear→sRGB
   (fs:139).

**Key takeaway:** the ship's directionality is the *baked per-tile dominant
direction*, not a per-light computation. All occlusion/colour/falloff physics is
already in `incoming_rgb` and `light_dir_2d`. The normal map only decides how the
surface catches that one incoming direction.

---

## 3. How are the light SOURCES represented?

**`bp.LightSource` is a plain value struct; the full per-frame source list is
assembled in `main.py` and handed to the renderer, but the renderer currently
consumes it only to build the field and does not retain it.**

C++ struct `LightSource` (`cpp/src/raycaster.h:53-71`), exposed via pybind
(`cpp/src/bindings.cpp:1253-1269`):

| field | type | default | meaning |
|---|---|---|---|
| `x`, `y` | float | — | **tile** coordinates |
| `max_range` | float | 20 | ray length in tiles |
| `ray_count` | int | 0 | 0 = auto from range+spread |
| `angle_center` | float | 0 | beam centre (rad) |
| `angle_spread` | float | 2π | 2π = omni, < 2π = cone/beacon |
| `intensity` | float | 1.0 | emitted power scale |
| `heat` | float | 0.0 | heat deposit (sim); level lights force 0 |
| `jitter` | float | 0.0 | per-ray angle RNG |
| `color` | float[3] | (1,1,1) | RGB tint (pybind property: get/set as tuple, `bindings.cpp:1266-1269`) |
| `falloff` | enum | UNIFORM | falloff profile |

**How many / static vs beacon (`main.py:302-331`):**
- `static_lights` — built ONCE from level.toml `[[light]]` entries that aren't
  beacons (`main.py:324-327`), via `light_source_params`
  (`src/level_lights.py:68`).
- `beacon_lights` — rebuilt EVERY frame from the sim tick (angle is a pure
  function of the monotonic sim tick so beacons freeze on pause / replay
  exactly, `main.py:377-387`, `level_lights.py:92-95`).
- Plus a mouse "flashlight" source built per frame (`main.py:388-399`:
  `max_range=25`, `intensity=2.5`, omni, cool-white `(1,1,1,0.95→)`).

Typical count is a handful (level lamps + optional beacons + 1 flashlight).

**Availability to the renderer each frame:** YES in principle — `main.py:402`
does `renderer.upload_state(sim.gmap, light_sources=sources)` with the fully
assembled list every frame. BUT `upload_state` (`game_renderer.py:275`) only
passes it to `compute_light_field` and **does not stash it** on the renderer.
Feeding real source positions/colours/radii to a marine shader (option b) would
require stashing that list (a trivial `self._light_sources = light_sources`).

---

## 4. How do sprites currently take light?

**A single scalar per unit, multiplied flat into the sprite/model tint.**

`_draw_units_world.light_at(u)` (`game_renderer.py:563-569`):
- samples `self.lighting.light_map` (the scalar brightness proxy from §1) at the
  unit's centre tile (`footprint//2` offset),
- adds `amb_floor` = mean of the shader's `u_ambient` RGB
  (`game_renderer.py:561-562`) so units in unlit rooms don't go pitch black
  while the ship keeps its ambient floor,
- returns `amb_floor + base` (an unbounded scalar; consumers clamp).

Sprite path: `draw_unit(..., light_intensity=light_at(m))`
(`game_renderer.py:600`) → `overlays.py:499-503`: `L = clamp(intensity,0,1)`,
`tint = (L*255, L*255, L*255, 255)` — a flat greyscale multiply of the whole
sprite. No direction, no colour (the lamp's RGB is discarded — a red lamp does
NOT redden a sprite, unlike the ship).

3D path mirrors this: `draw_units(..., light_fn=light_at)`
(`game_renderer.py:581-586`). In `unit_model_renderer.py:306-312`, `L =
clamp(light_fn(u),0,1)` and the per-unit `base_tint` RGB is scaled by `L` before
`DrawModelEx` — a flat whole-model multiply with the default raylib material
shader. So today the 3D marine is **flat-tinted**: correct overall brightness,
no directionality, no light colour, and it visibly mismatches the ship (which
has coloured, directional, normal-mapped light).

---

## 5. DESIGN RECOMMENDATION — how the marine shader should consume Breach's lights

### Recommendation: sample the ship's baked light field on the 3D mesh (a focused form of option c)

Give the marine a **custom material shader** that samples the SAME
`light_tex_a` / `light_tex_b` textures the ship uses, reconstructs
`incoming_rgb` + `light_dir_2d` exactly as `lighting.fs:107-111`, lifts the
direction to 3D with the SAME `u_light_z`, and does N·L against the **marine's
real geometric mesh normals** (no normal map needed — the mesh already has
them). Composite with the SAME `u_ambient`, `u_light_gain`, ACES tone-map and
sRGB encode. Keep the existing scalar `light_fn` path as the graceful fallback
when the shader is unavailable.

This is the parity-correct choice because **the marine then reads exactly the
field the ship reads**: same colour (a red lamp reddens both), same in-plane
direction, same occlusion (a marine in a shadowed/unlit room darkens because the
field is already zero there), same ambient floor, same tone-map. The marine's
advantage over the ship is that it has *true* normals from the glTF, so its N·L
is real geometry, not a faked tangent-space normal map — it will actually look
lit in 3D while staying colour-consistent with the flat ship.

### Why not (a) cheap scalar (what we half-have)
`light_fn` already gives the correct overall brightness, but it throws away
light colour and direction. A flat-tinted 3D body reads as a cardboard cutout
next to the directionally-lit ship — the whole point of the 3D marine is lost.
Keep it only as the fallback.

### Why not (b) raw `LightSource` uniforms + in-shader Blinn-Phong per source
Tempting ("proper" per-source lighting), but it actively BREAKS scene-match:
- The `LightSource` list carries no occlusion — you'd light the marine *through
  walls* unless you also re-implement the raycaster's shadow test in the shader.
- It ignores the coloured-gas absorption and 1/r falloff the field already
  resolved — the marine would diverge from the ship in exactly the visible ways.
- It needs the source list stashed on the renderer (§3) and a per-source loop +
  attenuation model in-shader — real work to reproduce, worse-matching output.

Reserve (b) for a future "hero lighting" upgrade (e.g. a rim/spec highlight from
the single nearest lamp) LAYERED ON TOP of the field sample — not as the base.

### Concrete shape of the recommended shader

Vertex shader: transform the skinned vertex normal by the model matrix to get a
world-space normal `N` (the marine's world axes are X = x_wpx, Y = up, Z = y_wpx
per `unit_model_renderer.py:22-29`); pass world position so the fragment can
derive the field UV.

Fragment shader (mirror `lighting.fs`):
- `world_uv = vec2(world_x / world_px_w, world_z / world_px_h)` — sample
  `u_light_a`/`u_light_b` at the marine's floor position. (Feed `world_px_w/h`
  from `WorldComposite`, `world_composite.py:37-38`.)
- `incoming_rgb = tex_a.rgb; dir2d = vec2(tex_a.a, tex_b.a);`
- **Axis map:** the ship's `light_dir.x`→world X, `light_dir.y`→world Z,
  `u_light_z`→world Y (up). So `L = normalize(vec3(dir2d.x, u_light_z,
  dir2d.y))` in the marine's 3D world frame. (The ship uses
  `vec3(dir2d, u_light_z)` because its tangent frame has Z up; the marine's
  world has Y up — same vector, reordered. Keep `u_normal_y_sign` available for
  the normal-map upgrade path so both conventions stay identical.)
- `ndotl = max(dot(N, L), 0.0);`
- `lit = albedo * (u_ambient + incoming_rgb * u_light_gain * ndotl);` then ACES
  + sRGB (copy `srgb_to_linear` / `aces_tonemap` / `linear_to_srgb` verbatim
  from `lighting.fs:65-78`).

`albedo` is the marine's group tint (`base_tint`, green marines / red zombies)
optionally × its diffuse texture — keep the marine/zombie colour identity.

### Reuse opportunities (all already present)
- **The field textures themselves** — `LightingPass.light_tex_a` /
  `light_tex_b` are public attributes already uploaded every frame
  (`lighting.py:65-66`, repacked `lighting.py:316-317`). Bind them on the marine
  material; no new GPU upload.
- **The scalar uniforms** — `LightingPass.ambient` (tuple, single source of
  truth, `lighting.py:133`), `.light_z` (`lighting.py:175`), `.light_gain`
  (`lighting.py:186`) are all cached on the pass for exactly this "other
  consumers read the same value the ship is lit by" reason
  (`lighting.py:130-133`). Push them straight into the marine shader.
- **The GLSL helpers + convention** — the sRGB and ACES functions and the
  `u_normal_y_sign` convention copy verbatim from `lighting.fs`, guaranteeing an
  identical tone/colour response.
- **`world_px_w/h`** — from `WorldComposite` (`world_composite.py:37-38`) for the
  UV mapping.

### Integration seam & cost
`unit_model_renderer._draw_one` is documented as the swap seam
(`unit_model_renderer.py:258-262`); setting a custom shader on
`model.materials[i].shader` and binding uniforms is a self-contained change
INSIDE this module. It is **orthogonal to CPU vs GPU skinning** — the material
shader runs on whatever skinned vertices `update_model_animation` produced, so
this can land before the GPU-skinning upgrade. Render-only, determinism-neutral
(the whole module is excluded from the synced sim/digest,
`unit_model_renderer.py:5-8`).

Tradeoffs to accept: (1) the field carries ONE dominant direction per tile, so a
marine between two lamps gets a blended direction — the *same* limitation the
ship has, which is the point (they match). (2) Bilinear sampling of a
physics-res field means smooth but not razor-sharp shadow edges on the marine —
again matching the ship. (3) One custom shader + per-frame uniform binding is the
only real cost; small and localized.

### Bottom line
Do NOT re-light marines from `LightSource` positions. Bind the ship's existing
baked light-field textures and cached ambient/gain/z uniforms to a small custom
marine material shader, and do real N·L on the marine's true mesh normals. The
marine inherits every physics decision the ship already made (colour, direction,
occlusion, falloff, tone-map) and simply catches that light on real 3D geometry.
Keep the current scalar `light_fn` as the fallback.
