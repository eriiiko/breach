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
- Result: a per-tile (or per-sub-tile) light map storing light intensity and color

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
- Characters and objects can cast shadows (optional, adds atmosphere)

### Integration with destructible environments
This is the key payoff: every time a wall is destroyed, the light map recalculates. A dark cargo bay suddenly floods with corridor light when a wall is blown open. This is emergent and dramatic with zero additional design effort — it falls out of the system naturally.

---

## 4. Smoke & Volumetric Light Interaction

- Dense smoke regions act as **semi-transparent occluders**
- Light passing through smoke is attenuated (dimmed) rather than fully blocked
- Creates volumetric-feeling light shafts where light enters a smoky room through a doorway
- Smoke density varies over time (from fire system, grenades, etc.) → light interaction is dynamic

### Visual effect
Light beam visible in smoke (god rays / volumetric light approximation). Can be done as a post-process or by rendering light intensity along the ray path where smoke density > 0.

---

## 5. Shadow Stealth Mechanic

Simple rule built on top of the lighting system:

- Sample the light intensity at the character's tile position
- If intensity < stealth threshold → character is **"in shadow"** / hidden
- Enemy AI visibility checks factor in light level at target position
- Creates tactical decisions: shoot out lights, use smoke, exploit dark rooms

This is a gameplay mechanic that emerges from the graphics system with minimal additional code.

---

## 6. Normal Maps for 2D Sprites

### What they are
A normal map is a texture where each pixel stores a **surface direction** (the normal vector) instead of a color. When combined with a directional or point light, the shader calculates how much light each pixel receives based on its facing direction. Surfaces facing the light brighten; surfaces facing away darken. The result: flat 2D sprites appear to have 3D depth and react to light angle.

### Related terms
- **Normal map:** stores surface direction per pixel (RGB = XYZ normal). This is what we use.
- **Height map:** grayscale image representing depth/elevation. Can be auto-converted to a normal map.
- **Bump map:** older term, similar concept to normal maps but implemented differently. People use the terms loosely.

### Production pipeline
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

## 7. Art Asset Strategy

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

## 8. Related Documents

- **Document 00: Architecture Overview** — engine choice discussion, two-layer architecture, full document roadmap
- **Document 07: Line of Sight & Cover** — the shadow stealth mechanic (Section 5 above) feeds directly into the LOS system
- **Document 05: Smoke Propagation** — smoke density drives the volumetric light interaction (Section 4 above)

---

*"Simple assets + rich systems + dynamic lighting = atmosphere that punches way above its weight."*
