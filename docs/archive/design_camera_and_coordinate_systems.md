# Design Note: Camera and Coordinate Systems

**Status:** open question, to be designed before next renderer changes.

## The problem

Today's rendering has at least three coordinate systems that need to coexist:

1. **World space** — tiles, integer (x, y), 0..grid_w × 0..grid_h
2. **Art space** — pixels of the diffuse texture (e.g., 1000×2400 for unhcr_vessel)
3. **Screen space** — window pixels (e.g., 1280×720 with a 280px panel)

And several data layers, each conceptually living in **world space** but stored at different *resolutions*:

| Layer | Storage resolution | Lives in world space? |
|------|--------------------|-----------------------|
| Diffuse texture | art res (1000×2400) | yes |
| Normal map | art res | yes |
| Emissive mask | art res | yes |
| Bloom | art res | yes |
| Light field (intensity + dir) | physics res (50×120) | yes |
| Smoke field | physics res | yes |
| Fire field | physics res | yes |
| Wall mask | physics res | yes |
| Future: heat field, water depth, gas type, lightning paths, units, particles | various | yes |

**The bug we hit:** the lighting shader samples the light field at `fragTexCoord` (visible-window UV), not at the world UV. When the camera scrolls, the light field's UV stays anchored at world (0,0) — flashlight illuminates the wrong place.

The same bug will hit smoke, fire, every future overlay, every shader that mixes textures at different anchors. Fixing it in N places means N opportunities to drift out of sync.

## What we want

A single, clean way to:
1. Tell a renderer "here is the visible region in world coordinates"
2. Have every layer (regardless of storage resolution) sample its world-space data at the right place automatically
3. Convert mouse position to world tile (already works correctly today)
4. Convert world tile to screen pixel (for drawing units, waypoints, particles)

## Approaches to consider

### A) Camera object that owns the visible region

```python
class Camera:
    camera_x: float          # in tile units
    camera_y: float
    fine_tile_px: float      # zoom

    def world_to_screen(self, fx, fy) -> (px, py): ...
    def screen_to_world(self, px, py) -> (fx, fy): ...
    def visible_world_rect(self) -> (x, y, w, h): ...   # in tile units
    def view_uv_for_layer(self, layer_w, layer_h) -> (u0,v0,u1,v1): ...
```

Every renderer module asks the camera for the source rectangle when drawing a world-space texture. Every shader receives `u_view_uv_offset` and `u_view_uv_size` uniforms set from the camera.

**Pro:** explicit, easy to reason about.
**Con:** every draw call needs to know it's a "world layer" and ask the camera.

### B) World-space render target

Render everything (diffuse, lit ship, smoke, fire, units, particles) into an off-screen texture at a chosen world resolution (e.g., grid_w × grid_h × tile_size_px). Then blit that texture to the screen with the camera transform applied to the blit.

```
                  +--- screen ---+
world-space RT -->|              | (one blit with camera src rect)
                  +--------------+
```

**Pro:** all layers automatically composite at world coordinates. Camera is just the final blit. Trivial to implement post-processing later (bloom, vignette, screen-space effects).
**Con:** allocates a large render target (e.g., 50×120 × 24px = 1200×2880 = 13.8 MB at RGBA8). Negligible on modern GPUs.

### C) Compositional approach with named coordinate spaces

A bit of both. Types/conventions in the renderer make it impossible to mix coordinate systems by accident:

```python
WorldPos = NewType('WorldPos', tuple)   # (fx, fy) integer tile
ScreenPos = NewType('ScreenPos', tuple) # (px, py) integer pixel
ArtPos = NewType('ArtPos', tuple)       # (px, py) art-texture pixel
```

Functions explicitly accept and return these. Camera transforms between them.

**Pro:** prevents mistakes, makes intent clear in code.
**Con:** verbose, may feel ceremonial in Python.

## My instinct

**Option B (world-space render target) is the cleanest** — it solves the coordinate problem once and for all by making every layer's compositing happen in world space, and the camera becomes a trivial single transform applied at the very end. It also unlocks future work for free:

- Post-processing (bloom, color grading, fog, vignette): one full-screen pass after the camera blit
- Screen capture / replay: dump the world-space RT to disk each frame
- Multiple cameras (split screen, security camera view): re-blit the same RT with different transforms
- Zoom in/out: change camera scale without affecting compositing

**The cost is one extra render-target allocation** (~14 MB), which is trivial.

## Open questions for Erik

1. Do you have a preference between A, B, C, or a fourth approach?
2. What's the largest world we might want to render at full resolution into a single RT? (50×120 is fine. A future 500×500 ship at 24 px/tile = 12000×12000 px = 576 MB. That would be too much — we'd need tiled rendering or render at lower zoom.)
3. Should the camera support arbitrary zoom (zoom in/out with mouse wheel)? Or fixed zoom levels?
4. Do we want screen-space post-processing layers in v1 (vignette around flashlight cone, etc.) or defer?

## Once decided

When Erik picks an approach:

1. Build the `Camera` class (or RT setup) cleanly
2. Refactor `renderer/game_renderer.py` to use it
3. Refactor lighting shader uniforms accordingly (or remove them if going RT approach)
4. Verify with a "scroll across the ship while flashlight follows mouse" test

This is the foundation for everything visual going forward — worth doing right.
