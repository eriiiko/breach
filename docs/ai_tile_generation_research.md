# AI-Assisted Level Art Generation for Breach

> **Date**: 2026-04-04 (research), 2026-04-05 (revised, first test generation)
> **Status**: Pipeline works end-to-end, first test image generated, needs iteration
> **Context**: Breach levels are unique ship interiors — each one a painting, not
> a repeated tileset. We need to generate beautiful ship art that respects the
> game's tile grid (1/3m tiles, walls >= 1 tile deep, corridors >= 3 tiles wide).
> Manual generation via ChatGPT produced nice images but gave no control over
> grid alignment, layout, or specific room content.

---

## The Problem (What We Learned)

Our first attempt at level art used ChatGPT image generation. The results looked
nice but:
- **No grid alignment**: Walls appeared between tile boundaries, breaking physics
- **No layout control**: Couldn't specify "put headless bodies in this room"
- **Time-consuming**: Fixing grid alignment manually was slow and still imperfect
- **Not repeatable**: Each generation was a roll of the dice

We need a pipeline that:
1. Takes our material grid as input (the layout is already defined by gameplay)
2. Generates art that respects tile boundaries
3. Gives creative control over room contents
4. Produces consistent style across a level
5. Outputs diffuse textures (normal maps generated separately via Laigter)

---

## Current State of Breach's Map System

- **Material grid**: `numpy` array of shape `[fine_h, fine_w]` with material IDs
  (air=0, hull=1, wood=2, door=3, steel=4, glass=5)
- **Rendering**: Currently solid color rectangles per coarse tile (no textures)
- **Prototype exists**: `prototypes/ship_to_level.py` converts ship PNGs to
  material grids (not yet integrated into game loop)
- **Ship images exist**: `frigatte0.png`, `frigatte1.png`, `science-vessel.png`,
  `art/ships/` — but none are loaded by the game yet
- **Grid**: 120x75 fine tiles (test map), 10px per fine tile currently
- **Graphics resolution**: Separate from game state resolution; not yet decided.
  Will be higher than the physics grid.

---

## Two Approaches (Both Needed)

We will pursue both approaches — they serve different purposes and complement
each other.

### Approach A: "Level as a Painting" (ControlNet + SDXL)

Each Breach level is a unique ship. ControlNet takes a spatial conditioning
image (derived from our material grid) and generates art that should respect
that layout. The whole level is one generated image.

```
Material grid (numpy)
    ↓
Render as edge image or segmentation mask
    ↓
ControlNet conditioning image
    ↓  + text prompt (sci-fi, horror, pixel art, etc.)
    ↓  + IP-Adapter style reference (optional, for consistency)
SDXL generates ship interior image
    ↓
Post-process: slice into tiles, verify grid alignment
    ↓
Laigter: generate normal maps from diffuse textures
    ↓
Game loads diffuse + normal map per tile (or as full-level images)
```

**Strengths**: Each level is unique. Rooms can have individual character.
**Weaknesses**: Getting the AI to respect fine grid details (corridors, thin
walls) is hard — our first test showed this.

### Approach B: Content-Aware Wang Tiles (for controlled tile-based art)

Generate sets of tiles that seamlessly connect. Useful for:
- Explicitly getting specific parts of levels right (e.g., a corridor section)
- Repeating material textures (hull plating, floor panels, wood grain)
- Areas where grid alignment is critical and must be pixel-perfect
- Building a library of reusable tile sets per environment type

**Strengths**: Guaranteed grid alignment. Reusable across levels. Precise control.
**Weaknesses**: Less unique — rooms can look repetitive without variety.

### Combined Workflow

The best results will likely come from combining both:
- Approach A for the overall level atmosphere and unique rooms
- Approach B for areas where precise grid alignment matters
- Manual touch-up where needed
- Laigter for all normal map generation (separate from AI art generation)

### Supporting Tools

- **GLIGEN**: Bounding-box control for placing specific objects in specific
  locations ("headless bodies in the cargo bay", "control panels on the bridge")
- **IP-Adapter**: Style consistency — generate one tile/image you like, use it
  as reference for all subsequent generations
- **CtrLoRA**: Train a custom Breach-specific control adapter cheaply if the
  off-the-shelf ControlNet models don't respect our grid well enough

---

## Layered Graphics Architecture

Breach's graphics use a layered approach, like Photoshop layers. This is
fundamental to gameplay: destroying a layer reveals what's underneath. Blowing
a hole in the hull exposes space. Breaking an interior wall shows the room
behind it. The physics simulation (decompression, fire, smoke) operates on
the material grid, and the visual layers reflect the state of that grid.

### Layer Stack (bottom to top)

| # | Layer | What it shows | Generation | Notes |
|---|-------|--------------|------------|-------|
| 0 | Space | Stars, nebulae, void | One large static image | Always exists behind everything |
| 1 | Ship exterior | Outer hull of the ship | One large AI-generated image | Does NOT need to align with interior |
| 2 | Hull structure | Engine details, structural beams | Tiling textures or painting | Revealed when outer hull is breached |
| 3 | Interior floor | Floor panels, grating, carpet | Tiling or per-room painting | Height map drives fluid simulation |
| 4 | Interior walls | Room walls, doors, furniture | ControlNet painting or Wang tiles | The main gameplay layer |
| 5 | Overlays | Fire, smoke, light, effects | Physics simulation (real-time) | Not pre-generated — computed live |

### How Destruction Works with Layers

When a wall tile is destroyed (`destroy_wall(fy, fx)`):
- The material grid changes (hull/wood → air)
- The renderer stops drawing that tile from its layer
- The layer below becomes visible at that position
- Physics responds (atmosphere vents, smoke flows through, light floods in)

**Example chain**: Explosion destroys hull tile → layer 1 (exterior) no longer
drawn at that position → layer 0 (space) shows through → atmosphere system
detects vacuum → air vents → smoke gets sucked out. All emergent, zero scripting.

### Simplifications

- **Exterior/interior alignment**: The exterior image does NOT need to match
  the interior layout geometrically. We can stop drawing the exterior entirely
  when the player is inside the ship, or scale it up so it always covers the
  level. This saves enormous complexity.
- **Ship exterior**: Could even be a real photograph or painting of a spaceship,
  since it's just a backdrop. Only the interior needs to respect the grid.
- **Multi-floor levels**: Each floor is a separate level — separate material
  grid, separate art layers. Staircases trigger level transitions (load new
  level or switch which level to render). Same engine, just different data.
  Supports scenarios like "fire in a tall building, escape down many floors."

### Per-Layer Generation Strategy

| Layer | Best tool | Why |
|-------|----------|-----|
| Space background | Any image generator, or photograph | No constraints |
| Ship exterior | SDXL (no ControlNet needed) | One-shot, no grid alignment |
| Hull structure | Content-Aware Tiles | Repeating metal/structural textures |
| Interior floor | Content-Aware Tiles + height map | Tiling, drives fluid sim |
| Interior walls | ControlNet (painting) + Wang tiles (precise areas) | Both approaches |

### Height Maps for Fluid Interaction

The fluid simulation (§6.8 in architecture.md) runs at pixel resolution using
a height map as terrain. This means:
- The floor layer's **height map IS the fluid terrain**
- Water reacts to every detail — drain grooves, raised thresholds, furniture
- The normal map is derived from the height map (used for lighting)
- AI-generated floor textures should be accompanied by height maps
- Laigter can generate height maps from diffuse textures (grayscale depth)

---

## Key Papers

### Tier 1 — Our Primary Tools

**ControlNet (ICCV 2023)**
- Paper: https://arxiv.org/abs/2302.05543
- Spatial conditioning (edge maps, segmentation masks, sketches) for diffusion.
  Our material grid rendered as a color map IS a segmentation mask.
- Available in diffusers, ComfyUI. No training needed.

**Content-Aware Tile Generation (SIGGRAPH Asia 2024)**
- Paper: https://arxiv.org/abs/2409.14184
- Code: https://github.com/samsartor/content_aware_tiles
- Training-free Wang tile generation via inpainting. For repeating material
  textures. Generates at 256px (SD2) or 512px (SDXL).

**Tiled Diffusion (CVPR 2025)**
- Paper: https://arxiv.org/abs/2412.15185
- Code: https://github.com/madaror/tiled-diffusion
- Seamless tileable images. Good for hull plating, continuous textures.

### Tier 2 — Compositional Control

**GLIGEN (CVPR 2023)**
- Paper: https://arxiv.org/abs/2301.07093
- Code: https://github.com/gligen/GLIGEN
- Bounding-box grounding for placing specific objects in specific locations.

**CtrLoRA (ICLR 2025)**
- Paper: https://arxiv.org/abs/2410.09400
- Code: https://github.com/xyfJASON/ctrlora
- Train custom ControlNet-like adapters cheaply (~1000 pairs, <1 hour, 1 GPU).
  Could train a Breach-specific "material map → ship art" control type.

**Canvas-to-Image (Snap Research, 2025)**
- Paper: https://arxiv.org/abs/2511.21691
- Code: https://github.com/snap-research/canvas-to-image
- Paint a rough canvas, get a polished image. Intuitive for artists.

### Tier 3 — Style Consistency

**IP-Adapter (2023)**
- Code: https://github.com/tencent-ailab/IP-Adapter
- Feed one reference image → all subsequent generations match its style.
  Critical for visual consistency across an entire ship.

### Tier 4 — Layout + Art Together (Future)

**WaveFunctionDiffusion**
- Code: https://github.com/wdcqc/WaveFunctionDiffusion
- WFC + SD for generating tile maps from text. Could eventually generate
  both layout and art simultaneously.

**Boris the Brave: Non-Manifold Diffusion (Feb 2025)**
- Blog: https://www.boristhebrave.com/2025/02/04/generating-tilesets-with-stable-diffusion/
- Practical tileset generation where all tiles connect coherently.

**Moonshine (AAAI 2025)**
- Paper: https://arxiv.org/abs/2408.09594
- Text-to-game-Map via diffusion. Could inspire text-driven level design.

### Surveys and Datasets

- PCG via Generative AI Survey: https://arxiv.org/abs/2407.09013
- GameTileNet (2,142 labeled tiles): https://arxiv.org/abs/2507.02941
- GAN Level Gen Survey: https://link.springer.com/article/10.1007/s11042-025-20612-9

---

## Tools Downloaded

All cloned to `breach/tools/`:

| Tool | Purpose | Repo |
|------|---------|------|
| content_aware_tiles | Wang tile generation (repeating textures) | samsartor/content_aware_tiles |
| tiled-diffusion | Seamless tileable image generation | madaror/tiled-diffusion |
| GLIGEN | Bounding-box compositional control | gligen/GLIGEN |
| IP-Adapter | Style consistency across generations | tencent-ailab/IP-Adapter |
| ComfyUI | Visual node-based SD pipeline | comfyanonymous/ComfyUI |
| ComfyUI-seamless-tiling | Tiling nodes for ComfyUI | spinagon/ComfyUI-seamless-tiling |
| seamless-tile-inpainting | Seam fixing extension | brick2face/seamless-tile-inpainting |
| CtrLoRA | Cheap custom control adapters | xyfJASON/ctrlora |
| ControlAR | Spatial control for AR models | hustvl/ControlAR |
| Canvas-to-Image | Paint-to-image generation | snap-research/canvas-to-image |
| WaveFunctionDiffusion | WFC + SD map generation | wdcqc/WaveFunctionDiffusion |
| TextureLab | Procedural texture generator | njbrown/texturelab |
| resynth-tiles | GIMP tileset tool | BorisTheBrave/resynth-tiles |

**SDXL model**: `tools/models/sd_xl_base_1.0.safetensors` (6.5 GB, downloaded)

**Also available**: Laigter (at `../laigter`) for normal map generation.

---

## Hardware

- Home PC (this machine): RTX 3070, 8 GB VRAM — runs SDXL
- Work PC: 12 GB VRAM — comfortable for SDXL + multiple ControlNets
- PyTorch 2.6 + CUDA 12.4 in base conda env (verified working)

---

## Implementation Plan

### Phase 1: Proof of Concept (First Session)

1. **Write a script** that renders Breach's material grid as a color-coded PNG
   (same colors as `_draw_map()` but saved to file)
2. **Run ControlNet + SDXL** with that PNG as conditioning + a sci-fi prompt
3. **Evaluate**: Does the output respect wall positions? Does it look good?
4. **Run Laigter** on the output to generate a normal map
5. **Display in Breach**: Load the generated image as background instead of
   solid rectangles (minimal game.py change)

### Phase 2: Refinement

- Add GLIGEN for room-specific content ("bodies here", "consoles there")
- Add IP-Adapter for style locking
- Generate repeating material textures via Content-Aware Tiles
- Iterate on prompts, ControlNet strength, resolution

### Phase 3: Pipeline Integration

- Automate: material grid → art generation → normal maps → game-ready assets
- Level editor workflow: design layout → generate art → tweak → export
- Multi-layer support (exterior, hull, interior, floor)

### Phase 4: Multi-Floor & Polish

- Multiple floors per ship (staircase transitions)
- Height maps for fluid simulation interaction
- Final art style decisions
- Asset caching and management

---

## First Test Results (2026-04-05)

**Script**: `prototypes/generate_test_level_art.py`
**Output**: `prototypes/test_level_generated.png`

### What We Did

1. Generated a random ship layout (40x25 coarse tiles, hull border, random
   wood walls with doors)
2. Rendered the material grid as a canny edge image (white edges on black)
3. Fed to ControlNet (canny variant) + SDXL with a sci-fi prompt
4. Generation took ~10 minutes on RTX 3070 (8GB VRAM) at 960x600

### What Worked

- The pipeline runs end-to-end: material grid → conditioning image → AI art
- The hull border (outer rectangle) was respected in the output
- The overall aesthetic is consistent (industrial/sci-fi)
- SDXL + ControlNet loads and runs on 8GB VRAM with CPU offloading

### What Didn't Work

- **Interior walls were mostly ignored.** The canny edges for wood walls and
  doors barely influenced the output. The generated image looks like one or
  two large rooms rather than the multi-room layout we specified.
- **Style was "warehouse" not "spaceship."** The prompt drove it toward
  industrial/factory aesthetics. Needs prompt tuning.
- **Grid alignment not visible.** Hard to see any tile grid in the output.

### Likely Causes and Next Steps

- **Canny ControlNet may be wrong tool.** Canny edges are thin white lines —
  fine interior walls produce very subtle conditioning. A **segmentation
  ControlNet** (color-coded regions) might work better since our material map
  already IS a segmentation mask.
- **ControlNet conditioning_scale (0.7)** might need tuning. Higher = more
  rigid adherence to edges, lower = more creative freedom.
- **Prompt engineering needed.** "2D game tilemap" or "pixel art top-down
  sprite sheet" might anchor SDXL to the right domain.
- **Resolution**: 960x600 is unusual for SDXL (trained on 1024x1024). Might
  get better results at standard resolutions, then crop/resize.
- **Consider training a CtrLoRA adapter** specifically for our material map →
  ship art mapping if off-the-shelf ControlNets can't handle it.

---

## Second Test Results — Hand-Painted Map (2026-04-06)

**Input**: `levels/Ship_walls_mats_test_20260406.png` — a hand-painted ship in
the Breach material palette (1 pixel = 1 tile, 112×112). Photopea palette file
provided at `art/breach_materials.aco`.

**Pipeline improvements**:
- Crop to bounding box of non-air pixels (much better resolution per tile)
- Auto-pick scale factor to fit ~1024×1024 (SDXL native size)
- Cropped to 81×41, scaled 12x → 488×968 output (much sharper than first test)

### Run A: Canny ControlNet — `prototypes/level_variations/`

5 variations at strengths 0.7, 1.0, 1.0, 1.0, 1.3 (different seeds).

**What worked:**
- **Ship outline clearly recognizable** in all 5. The dome at top, central
  spine, bottom appendage — all visible. The hand-painted layout has cleaner
  geometry than the random layout, which the AI handled much better.
- Style is consistent — a top-down sci-fi tile floor look.
- 5 minutes per variation at 488×968 (down from 10 minutes at 960×600).

**What didn't work:**
- Interior walls (compartment dividers) still ignored — the model painted
  decorative elements where the walls were instead of solid walls.
- Look is more "wooden tavern floor" than spaceship — prompt issue.

### Run B: Segmentation ControlNet (SargeZT/sdxl-controlnet-seg) — `prototypes/seg_variations/`

Same input, segmentation model instead of canny.

**What worked:**
- **Much better atmospheric quality.** Dark sci-fi corridors, blue accent
  lighting, looks like an actual top-down spaceship interior. Variation 4
  (seed 7) is particularly striking.
- More "dungeon map" / "facility floor plan" style.

**What didn't work:**
- **Completely ignored the ship outline.** Filled the entire rectangle with
  rooms; the dome and bottom appendage are gone.
- Reason: SargeZT was trained on **ADE20K segmentation colors** (a specific
  150-class palette). Our colors don't map to any of those classes, so the
  model treats the input as an abstract guide rather than a strict mask.

### Key Insight

We have two complementary failures:
- Canny respects geometry but produces wrong style
- Segmentation produces right style but ignores geometry

Both are tools we can fix — see "Pipeline Ideas" below.

---

## Pipeline Ideas (2026-04-06)

The right pipeline is probably not "one big generation" — it's compositional.
Below are ideas to explore.

### Idea 1: ADE20K Color Mapping

The SargeZT segmentation model expects ADE20K class colors. We can build a
mapping from Breach materials to the closest semantically-meaningful ADE20K
classes, then convert our material map before feeding to the model.

| Breach material | Likely ADE20K class | ADE20K class color |
|---|---|---|
| Air (interior floor) | floor (#50,#50,#50) | TBD |
| Hull | wall (#120,#120,#120) | TBD |
| Wood | wall (different shade) | TBD |
| Door | door class | TBD |
| Steel | wall | TBD |
| Glass | window class | TBD |

Either build this mapping ourselves or just copy the ADE20K palette. Could
unlock the segmentation model's full power.

### Idea 2: Per-Region Generation (room-by-room)

Instead of generating the whole ship at once, segment it into rectangles and
generate each separately, then composite:

1. Detect rooms in the material map (flood-fill connected air regions)
2. For each room: extract bounding rectangle
3. Generate art for that room with a room-specific prompt:
   - "cockpit, control panels, captain's chair"
   - "biology lab, glass tanks, specimens"
   - "living quarters, bunks, lockers"
   - "engineering, reactors, pipes"
4. Generate corridors separately with a corridor-specific prompt
5. Composite all generations into a single image
6. Blend seams (feather edges, or use SD inpainting to smooth transitions)

Benefits:
- Each room can have its own purpose and aesthetic
- Smaller images = faster generation, more detail per tile
- Can iterate on individual rooms without regenerating the whole level
- Naturally produces variety

Challenges:
- Seam blending between rooms
- Maintaining consistent global style (use IP-Adapter with fixed reference)
- Decoration must respect game logic (no walls inside an air region)

### Idea 3: Layer Stitching (floor first, then walls)

The layered architecture (see "Layered Graphics Architecture" above) suggests
generating each layer independently and stitching them:

1. **Floor layer**: Generate as one large image, OR stitch from per-region
   floor segments. Floor is mostly continuous and can be more relaxed.
2. **Wall layer**: Generate separately, possibly using **content-aware Wang
   tiles** (the technique we discussed earlier). Walls are constrained to the
   tile grid and need to be pixel-perfect.
3. **Furniture/decoration layer**: Generated per-region, placed on top of floor.
4. **Composite**: Walls on top of floor, furniture on top of walls, etc.

This is closer to how a game artist would build a level by hand.

### Idea 4: Wall Mask Post-Processing

Regardless of how we generate the art, we can ALWAYS guarantee wall correctness:
1. Generate art with any method
2. Apply the original material map as a hard mask
3. Where the material map says "wall", overlay a wall texture (or darken)
4. Where it says "air", show the generated art

This is the safety net — even if the AI invents a corridor where there
shouldn't be one, the post-processing fixes it. Walls always land on the grid.

### Idea 5: Furniture as Heightmap (Climbable Geometry)

Furniture could be generated as part of the floor art, then a height map
extracted from it (Laigter does this from grayscale). The game then treats
anything above a height threshold as "climbable" rather than "passable":
- Players can climb on tables, crates, beds
- Aliens use them differently (jumping AI)
- Fluids pool around them (already supported by the pipe model)
- Fire spreads over them
- No need for explicit "this is a table" entities

This is elegant: the same height map that drives the lighting normal map
ALSO drives the gameplay collision and fluid simulation.

### Idea 6: Iterative Workflow

Whatever pipeline we land on, we need to be able to iterate:

```
Design layout in Photopea (1px = 1 tile)
    ↓
Save material map PNG to levels/
    ↓
Run pipeline → preview
    ↓
Tweak prompt / regions / seed
    ↓
Re-run only the parts that changed
    ↓
Final assets baked to game-ready files
```

The "re-run only what changed" part is critical for iteration speed.
Per-region generation supports this naturally — change one room, regenerate
just that room.

### Pipeline Sketch (Combining the Ideas)

```
1. Designer paints material map in Photopea (palette: art/breach_materials.aco)
2. Tool detects rectangular regions (rooms) and corridors
3. Designer can label each region (cockpit, lab, etc.) — optional
4. For each region:
     a. Generate floor + furniture art (region-specific prompt + IP-Adapter)
     b. Extract height map via Laigter
     c. Cache result, keyed on region content + prompt
5. Composite floors into one big image (with feathered seams)
6. Generate or look up wall textures (Content-Aware Wang tiles)
7. Stamp walls onto the composite (hard mask from material map)
8. Generate normal maps from final composite via Laigter
9. Export to game-ready assets:
     - diffuse.png
     - normal.png
     - height.png
     - material_map.npy (the original layout)
10. Game loads these and renders
```

This is a sketch — not implemented yet. Each step has subtleties to work out.

---

## Open Questions

- **Graphics resolution**: Not yet decided. Higher than physics grid. Needs to
  be chosen — affects ControlNet output size, tile texture size, performance.
  Suggestion: start with 32x32 px per fine tile (3840x2400 for test map) and
  see how it looks.
- **How to handle destroyed walls**: When a wall is destroyed, what's behind it?
  Pre-generate the "revealed" layer? Or dynamically switch to a lower layer?
- **Art style**: Pixel art leaning, with normal maps for lighting depth. Need
  reference images to lock the style with IP-Adapter.
- **Region detection**: How do we detect rooms in a material map? Simple
  flood-fill of air regions, then bounding box? Or smarter rectangle
  decomposition?
- **Region labelling**: Should the designer label rooms manually, or should
  we use an LLM to look at a room's shape/position and suggest a purpose?
- **Furniture authoring**: Is furniture generated by the AI as part of the
  room art, or placed manually as separate sprites? Or both?
- **Seam blending**: How do we blend the edges between independently-generated
  regions? Feathering? SD inpainting on the seams? Both?
