# Graphics & Lighting System Design
## Breach — Document 13: Graphics, Lighting & Art Pipeline

*Distilled from conversation, Feb 28 2026*
*Part of the Breach design document series (see Document 00: Architecture Overview)*

---

## 1. Design Philosophy

Low-fidelity art assets + rich systems = high-impact visuals. The lighting and particle systems do the heavy lifting. Even placeholder sprites look atmospheric when dynamic light rakes across normal-mapped surfaces and shadows shift in real time as walls are destroyed.

**Priority:** Find low-hanging fruit with maximum visual impact. The lighting system is exactly that — cheap to compute, dramatic results, and it ties directly into existing gameplay mechanics (destructible walls, smoke, fire).

---

## 2. Tile & Resolution System

- **Base unit:** 1 tile (e.g., 32px × 32px — exact size TBD)
- **Player/character:** occupies 3×3 tiles → sprite is 3× base resolution (e.g., 96×96px)
- **Walls:** 1 tile deep → 32×32px texture
- **Corridors:** minimum 3 tiles wide (to fit one character)
- **Doors/frames:** sized to span, e.g., 1×3 tiles = 32×96px
- **Normal maps:** match sprite dimensions exactly

All assets snap to the tile grid. Destruction operates along tile boundaries, keeping both the simulation and the art pipeline clean.

---

## 3. 2D Raycasting Light System

**Not GPU ray tracing (RTX).** This is 2D raycasting — computationally cheap and well-suited to a tile-based game.

### How it works
- From each light source, cast rays outward across the 2D plane
- Rays terminate when they hit an opaque occluder (wall, closed door, solid object)
- Everything behind the occluder is in shadow
- Result: a per-tile light map storing light intensity, direction and color

### Light sources
- **Room lights:** static ceiling fixtures, update only when walls change
- **Helmet/weapon lights:** move with characters, cast dynamic moving shadows
- **Fire:** flickering point lights (use noise function for flicker)
- **Explosions:** brief intense flash that illuminates everything momentarily
- **Muzzle flash:** very short duration, dramatic in dark rooms
- **Emergency lighting:** red-tinted, activates on hull breach or alarm

### Shadow behavior
- Walls cast hard shadows
- Destroyed walls update the occlusion map → light floods into previously dark areas
- Doors opening/closing changes shadow geometry
- Characters and objects cast shadows

### Integration with destructible environments
This is the key payoff: every time a wall is destroyed, the light map recalculates. A dark cargo bay suddenly floods with corridor light when a wall is blown open. This is emergent and dramatic with zero additional design effort — it falls out of the system naturally.

### Color treatment: desaturation in shadow

Brightness and saturation are independent dimensions. Darkness should feel **colorless**, not merely dim. An earlier prototype achieved this with two texture versions — one full-color, one fully (or 90%) desaturated — blended by light intensity. The contrast was strong: shadowed areas read as drained, lit areas pop with color. This effect was lost in the renderer refactor (we need to confirm but i think so) and needs to be reintroduced.

Two viable implementations:
- **Two-texture blend** — keep both versions, blend by light intensity. (What the original did.)
- **Shader (preferred)** — desaturate at render time in the fragment shader. Pass in the light field / raycast result; lerp each pixel from full saturation to grayscale based on how lit it is. Cleaner than maintaining two diffuse textures, and makes the saturation-vs-brightness distinction explicit in one place.

### Implementation status: disable Raylib's built-in lights

After the renderer refactor, the custom raycast lighting works correctly, but Raylib's built-in lighting is also still active — everything is brighter than intended and the directional light gets washed out. **Decision:** disable Raylib's built-in lights and run on the custom raycast alone. If a hybrid is ever wanted, it can be layered back on deliberately rather than by accident.

### Per-source update strategy (light source class)

Each light source has a different natural update cadence. Recasting all sources every tick is wasteful when most of them haven't moved.

- **Flashlight / helmet lights:** recast every tick — they follow the character.
- **Muzzle flash, explosion flash:** one-shot, recast only on the trigger frame.
- **Static room lights:** recast only when occlusion changes (wall destroyed, door opens/closes). Otherwise cached.
- **Fires:** the ray geometry is static-ish for a few rounds at a time (fires move slowly), but moving characters cast shadows through firelight every tick. Open design question: can the per-source ray paths be cached and only the moving-occluder contributions recomputed each tick? Worth exploring.

This points toward a `LightSource` wrapper on the Python side that knows its own dirty state ("needs recast?") and a lighting pass that iterates only dirty sources. The C++ `LightSource` struct in [cpp/src/raycaster.h](../cpp/src/raycaster.h) already holds the geometry (position, range, angle, falloff, jitter); the Python wrapper would add the scheduling layer on top.

### Note on raycaster implementation (not GPU, not RTX)

The lighting raycaster is **2D CPU code in C++** ([cpp/src/raycaster.cpp](../cpp/src/raycaster.cpp), DDA ray marching), called from Python via pybind. Not GPU-accelerated despite the rest of the pipeline using GPU for rendering. CUDA-ing it is listed in [TODO.md](TODO.md) under "CUDA Migration" — it's embarrassingly parallel and a natural first kernel.

For clarity: **"RTX" specifically means NVIDIA's hardware-accelerated 3D ray tracing** (BVH traversal + ray-triangle intersection in dedicated silicon, for 3D scenes). Breach does 2D ray *marching* through a tile grid — same word "ray," entirely different machinery. The grid structure also makes this cheap in ways 3D ray tracing isn't.

---

## 4. Smoke & Volumetric Light Interaction

- Dense smoke regions act as **semi-transparent occluders**
- Light passing through smoke is attenuated (dimmed) rather than fully blocked
- Creates volumetric-feeling light shafts where light enters a smoky room through a doorway
- Smoke density varies over time (from fire system, grenades, etc.) → light interaction is dynamic

### Visual effect
Light beam visible in smoke (god rays / volumetric light approximation). Can be done as a post-process or by rendering light intensity along the ray path where smoke density > 0.

### Idea: normal-mapped smoke tiles

Smoke concentration lives at physics resolution (one value per tile), but the **visual** smoke pixels within a tile sit at game resolution. That opens the door to painting a normal map onto smoke — a "smoke normal" sub-texture that lets the smoke not just attenuate light, but also catch directional highlights from it.

Use the same per-pixel dot-product treatment as solid sprites: each smoke pixel gets a fake surface normal, dotted with the local light direction (already available in our packed light field, G/B channels). The result is internal shading inside the smoke volume — wisps, eddies, density variation reading visually instead of as flat gray fog.

Cheap to add since the renderer already samples light direction per pixel; the only cost is authoring or generating the smoke normal texture. Worth exploring once the basic lit-smoke + god-rays approach is in.

---

## 5. Shadow Stealth Mechanic

Simple rule built on top of the lighting system:

- Sample the light intensity at the character's tile position
- If intensity < stealth threshold → character is **"in shadow"** / hidden
- Enemy AI visibility checks factor in light level at target position
- Creates tactical decisions: shoot out lights, use smoke, exploit dark rooms

This is a gameplay mechanic that emerges from the graphics system with minimal additional code.
Also - PErhaps the AI's main input will be the physics grid or even the rendered image - so that if we have trouble seeing something, it also has trouble seeing something. Im not sure if it's the right way to implement htis . but perhaps it is very cool. (different species have different views of the gamestate, infravision etc)

---

## 6. Normal Maps for 2D Sprites

### What they are
A normal map is a texture where each pixel stores a **surface direction** (the normal vector) instead of a color. When combined with a directional or point light, the shader calculates how much light each pixel receives based on its facing direction. Surfaces facing the light brighten; surfaces facing away darken. The result: flat 2D sprites appear to have 3D depth and react to light angle.

### Related terms
- **Normal map:** stores surface direction per pixel (RGB = XYZ normal). This is what we use.
- **Height map:** grayscale image representing depth/elevation. Can be auto-converted to a normal map.
- **Bump map:** older term, similar concept to normal maps but implemented differently. People use the terms loosely.

### Production pipeline
0. Create the walls and rooms at physics resolution first. The pipeline is still in prototype phase as of 20260520.
1. Draw/obtain the base sprite (diffuse/color texture)
2. Create or auto-generate a height map (grayscale depth)
3. Convert height map → normal map (automated)
4. In-engine shader: for each pixel, dot product of (light direction) and (normal map value) = lighting intensity

### Tools for auto-generating normal maps
- **Laigter** — free, open source, purpose-built for 2D game sprites
- **SpriteIlluminator** — commercial, very polished
- **GIMP/Photoshop plugins** — workable for simpler needs
- **AI generation** — improving rapidly, viable for textures now

### In-engine implementation
Most 2D engines support this natively or with minimal shader work:
- Unity 2D Lights: built-in normal map support
- Godot: CanvasItem shaders with normal maps
- Custom engine: straightforward fragment shader (dot product per pixel)

---

## 7. Destruction Painting Layer

A single full-level "edit texture" is the substrate for **all** destructive visual changes — bullet holes, blood splatter, scorch marks, rubble, floor painted over destroyed walls. One texture, not one layer or sprite per effect type.

### 7.1 Core paint system

**Goal:** unlimited, effectively free destructive edits to the level without exploding sprite or layer counts.

**Approach:** one edit-layer texture sized to the level. All destructive effects are paint operations onto this texture; the original level PNG stays untouched.

- A handful of pixels per event is microseconds; even filling the screen with edits is fast.
- The real cost is VRAM for the texture — negligible for a single-level game.
- Raylib supports the texture manipulation directly.

**Decision:** single-layer, paint freely. Only optimize if a real bottleneck shows up.

### 7.2 Patch A — Grenade scorch marks via normal map

When a grenade explodes, for each pixel inside the blast radius:

1. Sample the normal map at that pixel.
2. Compute the dot product between the pixel's normal and the explosion direction (vector from blast center outward to the pixel).
3. Paint the scorch scaled by that dot product — surfaces facing the blast darken strongly; surfaces parallel to or facing away pick up little or no mark.

Directionally realistic burn marks fall out for free, reusing normal-map data the renderer already has. Same painting code path as the rest of the destruction layer, with a directional mask applied based on surface orientation.

### 7.3 Patch B — Destroyed wall tiles (quickfix)

When a wall tile is destroyed:
- Paint floor or rubble texture over those pixels so the tile reads as walkable.

-After thinking aobut this - i'm a litte intrigued. perhaps we can paint with texturs, and perhaps this vcan be done really really cool. Paint with assets is another possability, rubble etc.
- Straight replacement is fine for v1 — no animation needed.

(v2 idea: lerp between intact and destroyed states for a gradual crumble.)

Same texture-paint code path as scorch marks.

### 7.4 Future: doors as animated assets, not paint operations

The wall-destruction patch above is a quickfix. Long-term, **doors specifically should become first-class animated assets** — not tiles painted on the edit layer — so they can play open/close animations, hold per-door state, and interact properly with the occlusion grid for the lighting system. The paint-based approach stays for genuinely destructive edits (rubble, scorch, blood); animated/stateful tiles graduate to assets.

### 7.5 Blood splats — same paint layer, unit-driven source (Erik, 2026-05-23)

Blood splats reuse the destruction painting layer (§7.1). The mechanism is the same as scorch marks; only the trigger and the brush change:

- **Trigger**: a unit takes ranged damage, melee damage, or dies. Each such event paints onto the floor "behind" the unit relative to the source of the hit. A bullet's direction of travel determines the splatter direction.
- **Brush**: a small irregular blob of dark-red pixels with feathered edges and minor variance. Multiple hits stack and grow the stain.
- **Normal-map interaction**: blood doesn't relieve like scorch does — it's a stain, not a burn — so the normal-map contribution stays at zero (or very subtle wet sheen). The diffuse layer carries the entire effect.
- **Pooling around dying units**: a downed unit grows a spreading pool over a few seconds (could be a timed expansion of the stain centered on the unit's anchor tile).

Implementation note: same code path as `paint_scorch_at(tile, direction)` — add `paint_blood_at(tile, direction)`. The painting layer doesn't care what's painting; the brush is the variable.

---

## 8. Art Asset Strategy

### Phase 1: Prototype (now)
- Colored rectangles or ultra-simple shapes for characters
- Basic tiled textures for walls/floors (even solid colors work)
- Focus entirely on getting the simulation + lighting working
- **Goal:** light spilling through a doorway into a dark room, blow a wall open, watch shadows recalculate

### Phase 2: Placeholder assets
- Source open-licensed sprite packs (itch.io, OpenGameArt)
- Top-down or isometric sci-fi soldiers with basic walk/shoot/die animations
- Simple tileable metal/floor/wall textures
- Auto-generate normal maps from all textures

### Phase 3: AI-generated assets
- Static textures (floor panels, hull plating, wall surfaces) — viable now
- Tileable texture generation + auto normal maps
- Sprite generation with animation frames — improving rapidly
- Timeline works in our favor: engine first, better AI art tools by the time we need final assets

### Phase 4: Final art
- Hand-painted or professionally generated sprites with custom normal maps
- Consistent art style across all assets
- Animation polish

---

## 9. Related Documents

- **Document 00: Architecture Overview** — engine choice discussion, two-layer architecture, full document roadmap
- **Document 07: Line of Sight & Cover** — the shadow stealth mechanic (Section 5 above) feeds directly into the LOS system
- **Document 05: Smoke Propagation** — smoke density drives the volumetric light interaction (Section 4 above)

---

*"Simple assets + rich systems + dynamic lighting = atmosphere that punches way above its weight."*
