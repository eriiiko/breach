# Breach — 2D Graphics & Rendering Course
*Conversation transcript*

---

## Topic 1: The mental model — everything is layers

In 2D game graphics, almost all complexity is solved by **stacking layers with different blend modes and visibility rules**.

For Breach specifically, your ship cross-section probably needs these conceptual layers (bottom to top):

1. **Void/space** — what's behind everything, shows through holes
2. **Structural skeleton** — the damaged/destroyed state of the hull
3. **Hull/walls** — the intact ship structure (destructible)
4. **Interior floor** — floor tiles, stains, details
5. **Entities** — characters, objects, furniture
6. **Interior ceiling/overhead** — things above the action plane (pipes, lighting rigs)
7. **Lighting layer** — the flashlight and ambient light composited on top
8. **Emissive layer** — things that glow regardless of lighting
9. **UI** — HUD, etc.

The exterior view (before breaching) would be its own separate scene entirely — interior and exterior should be kept independent.

---

## Topic 2: Tilemaps and destructible walls

In Godot (and analogously in Raylib), a **TileMap** is a grid where each cell references a tile from a tileset. You can have multiple TileMap nodes stacked — one for floors, one for walls, one for decals, etc.

For destructibility:

- Don't actually destroy the tile visually in isolation. When a wall tile is destroyed, *replace* it with a "rubble" or "damaged" tile, and simultaneously reveal the skeleton layer underneath.
- The skeleton layer is a second TileMap sitting below the wall TileMap, always drawn but normally hidden behind intact wall tiles.
- Transparent holes to space: when a tile is fully destroyed, leave that cell empty on the wall TileMap. Since the void/space layer is at the very bottom, space shows through automatically.

Your skeleton TileMap just needs to be painted everywhere underneath the hull — you never have to dynamically generate it. The destruction logic is: *remove wall tile → skeleton tile was always there, now visible.*

---

## Topic 3: The lighting system

What you've built (dark desaturated image + fully lit original + flashlight reveal) is a legitimate **light masking / reveal system**. Named properly:

- Your **dark desaturated image** is the "unlit" state — the ambient darkness
- Your **fully lit original** is the "lit" texture
- The **flashlight** determines which region of the lit texture to reveal

### In Raylib

Since you're using Raylib + C++ (pure text-based, no node graphs), all lighting is implemented via GLSL shaders you write yourself.

The pre-baked dark version approach has an advantage: **precise artistic control** over how dark each material looks, including desaturation (stone goes grey-blue in darkness, not just dim). If that's important to the aesthetic, it's worth keeping and treating as a shader parameter.

### Emissive maps

An emissive map is exactly right for screens. Pixels in the emissive map are *added* on top regardless of the lighting pass.

- Draw your screen glow as a separate sprite on a layer **above** the lighting layer, with **additive blend mode**
- For the bloom/glow halo: render a blurred, slightly larger version underneath it
- The pre-baked approach (paint the glow halo as part of the art asset, render additively) is what many pixel art games do — simpler than runtime blur and looks great

---

## Topic 4: AI generation and layer alignment

The issue with generating interior and exterior separately and having them not align is fundamental: diffusion models have no concept of geometric consistency across separate generations.

### Single-prompt approach

Generate all three states simultaneously as a sprite sheet:

> *"Pixel art sprite sheet, 3 panels side by side, same spaceship cross-section: left panel = intact exterior hull view, center panel = interior room view same dimensions, right panel = destroyed skeleton/wreckage same dimensions. All three panels share identical geometry and proportions. Dark sci-fi palette."*

Key words: "same dimensions," "identical geometry," "sprite sheet" — these cue the model toward treating the three as views of one object. Then correct misalignments manually in Photoshop/Aseprite.

---

## Topic 5: Seamless tiles

When you generate a texture with AI (or paint one), the left and right edges don't match — tiling reveals a visible seam grid.

### The offset trick (Photoshop)

1. Take your texture
2. **Offset it by exactly 50% in both X and Y** (Filter → Other → Offset, use half the image dimensions, set to Wrap Around)
3. Now the seams are in the middle — a visible cross
4. **Paint over that cross** — clone stamp, heal, or manual paint
5. The result tiles seamlessly

### ComfyUI

You can set a "tiling" flag on the generation itself, so the model bakes seamlessness in during generation. Much cleaner. The Photoshop method works with any source.

---

## Topic 6: Normal maps — everything

### What they actually are

A normal map is a texture where each pixel encodes a **surface direction vector** (a "normal") rather than a color. That vector tells the lighting shader which way that spot on the surface is "facing."

| Channel | Encodes | Flat surface value |
|---|---|---|
| R | X (left/right tilt) | 128 (0.5) |
| G | Y (up/down tilt) | 128 (0.5) |
| B | Z (out of screen) | 255 (1.0) |

That's why normal maps have that characteristic **flat blue-purple color** — a surface pointing straight out of the screen is (0.5, 0.5, 1.0). Tilted surfaces shift toward red (tilting right) or green (tilting up).

### The lighting calculation

```glsl
vec3 normal = texture(normalMap, uv).rgb * 2.0 - 1.0; // unpack from [0,1] to [-1,1]
vec3 lightDir = normalize(lightPos - fragPos);         // direction toward light
float intensity = max(dot(normal, lightDir), 0.0);     // how directly facing the light
vec3 lit = diffuseColor * lightColor * intensity;
```

Extensions:
- **Ambient** — a minimum brightness so nothing is fully black
- **Specular** — a highlight based on the reflection angle
- **Attenuation** — light falls off with distance (usually `1 / distance²`)

### Large surfaces at an angle

Normal maps can encode any surface angle at any scale — not just small micro-detail patterns.

For an A-wing style wing tilted steeply toward the viewer, the entire wing face would have a **uniform normal color** corresponding to that angle — shifted toward green-red depending on tilt direction, not the flat blue-purple of a face-on surface.

What this gives you:
- When a light is positioned **in front** of the ship, the angled wing face catches it strongly — it brightens dramatically
- When the light moves to the side the wing was tilting away from, the face dims or goes dark
- This is indistinguishable from actual 3D geometry

You can **layer** macro and micro normal maps: the macro encodes the big wing angle, a tiling micro normal adds panel line and rivet detail on top. This technique is called **normal map blending**.

### Formula: angle to color

```
normalColor.r = (sin(tiltX) * 0.5) + 0.5
normalColor.g = (sin(tiltY) * 0.5) + 0.5
normalColor.b = cos(angle_from_screen) * 0.5 + 0.5
```

### Tools for creating normal maps

- **Laigter** — free, specifically designed for pixel art, excellent for sprites
- **SpriteIlluminator** — paid, more control
- **NormalMap-Online** — browser tool, quick and dirty
- Any of the above can convert a greyscale **height map** (bright = raised, dark = recessed) to a normal map automatically

### In Raylib

```c
Texture2D diffuse = LoadTexture("ship_diffuse.png");
Texture2D normalMap = LoadTexture("ship_normal.png");
Shader lightShader = LoadShader(0, "lighting.fs");

BeginShaderMode(lightShader);
    SetShaderValueTexture(lightShader, normalMapLoc, normalMap);
    SetShaderValue(lightShader, lightPosLoc, &lightPos, SHADER_UNIFORM_VEC2);
    DrawTexture(diffuse, x, y, WHITE);
EndShaderMode();
```

**Note:** Raylib's default coordinate system has Y pointing down. Normal maps are typically Y-up. If lighting looks inverted vertically, **flip the G channel** in your shader.

### Dynamic lighting payoff for Breach

- **Explosions** — a flash point light that reveals true geometry of everything in its radius
- **Flashlight** — as it sweeps, an angled wing face responds differently than a face-on wall
- **Muzzle flash** — single-frame point light, dramatic shape snap
- **Screen glow** — emissive screens cast colored light that interacts with surface normals on nearby walls

---

## Topic 7: What is a shader?

### The core distinction

| | Texture | Shader |
|---|---|---|
| What it is | Data (pixels) | Program (code) |
| Lives on | GPU memory | GPU execution units |
| Runs | Never — it's looked up | Once per pixel, every frame |
| Changes | Only if you upload new data | Every frame, different inputs |
| Analogy | A photograph | The person deciding how to print it |

**The texture is the raw material. The shader is the fabrication process.**

### The pipeline

When you draw something, the GPU runs two shaders in sequence:

1. **Vertex shader** — runs once per corner of your geometry. Transforms world-space positions to screen positions. For 2D sprite work, Raylib's default is usually fine.

2. **Fragment shader** (pixel shader) — runs once per pixel the geometry covers. Outputs one value: the final RGBA color of that pixel. This is where almost all 2D effects live.

If you draw a 200×150 sprite, the fragment shader runs **30,000 times** — once per pixel — all in parallel.

### What a fragment shader looks like

Simplest possible — everything red:

```glsl
void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
```

Passthrough — draw the texture as-is:

```glsl
uniform sampler2D texture0;
varying vec2 fragTexCoord;

void main() {
    gl_FragColor = texture(texture0, fragTexCoord);
}
```

With normal map lighting:

```glsl
uniform sampler2D texture0;
uniform sampler2D normalMap;
uniform vec2 lightPos;
uniform vec3 lightColor;
varying vec2 fragTexCoord;
varying vec2 fragPos;

void main() {
    vec4 diffuse = texture(texture0, fragTexCoord);
    vec3 normal = texture(normalMap, fragTexCoord).rgb * 2.0 - 1.0;
    
    vec3 lightDir = normalize(vec3(lightPos - fragPos, 0.5));
    float intensity = max(dot(normal, lightDir), 0.0);
    
    gl_FragColor = vec4(diffuse.rgb * lightColor * intensity, diffuse.a);
}
```

The exact same sprite, drawn with the exact same texture, looks completely different depending on where the light is. The texture didn't change — the *program interpreting it* changed.

### The parallel execution model

You don't write a loop. You write the body of what happens to **one pixel**, and the GPU runs thousands of copies simultaneously.

This is why shaders can't communicate between pixels. Your shader for pixel (100, 50) cannot ask what color pixel (99, 50) got — they're running in parallel. Blur effects that need neighboring pixels must sample the *texture* (previous frame's data), not adjacent shader invocations.

### Effects in Breach that map to shaders

- **Flashlight reveal** — compare pixel position to light position, interpolate between dark and lit samples
- **Emissive screens** — read emissive mask texture, add that value regardless of lighting
- **Normal map lighting** — as above
- **Hull damage overlay** — blend a damage texture based on per-tile health value passed as a uniform
- **Void/space shimmer** — subtle parallax or nebula animation on the void layer

The pattern is always the same: pass data in (textures, numbers, positions), the shader decides what color comes out.

---

*End of transcript*
