# Fire — Design Notes

> Living notes for the fire system. Pull in here whatever fire-related design
> thinking accumulates over time; promote to a proper design doc when scope
> warrants.

---

## Fire as a short-range light source (Erik, 2026-05-23)

When fire (a burning tile, a flamethrower stream, a torch) emits light, the question is *how* to integrate it with the existing raycaster without blowing up the cost or inventing exceptions.

**Decision**: each burning tile is a regular `LightSource` in the existing raycaster pipeline. **No special "fire light" code path.** The discipline: stick to the physics sim, no exceptions.

**Cost discipline via `max_range`**: a flamethrower stream might have ~10-30 burning tiles at once. If each casts long rays the total ray count explodes. The fix is per-source `max_range` set small for fire (probably 2–4 tiles). The raycaster's cost is proportional to total rays cast — many sources × few rays = same as few sources × many rays. Short-range fire sources are therefore cheap even when many tiles burn.

**Why this beats "paint a few tiles around the flame"**: the painted-tiles approach loses wall occlusion. Fire next to a wall would bleed light onto the wrong side. Short-range raycasting preserves the right behaviour with no cost penalty.

**Open implementation question**: at what range does a fire-tile light fall off, and how does intensity scale with fire-intensity (0..1)? Probably `range = 2 + 2 * fire_intensity` and `intensity = 0.3 + 0.7 * fire_intensity`. Tune in the lighting demo when fire-source emission is wired in.

**Related future system**: explosions emit light from pressure/temperature tiles — same idea, different source, see `memory/project_explosion_as_light_idea.md`.

---

## Open items (to flesh out in upcoming physics-engine pass)

- Wire fire tiles into the LightSource list (not yet done; main.py only includes mouse flashlight + static emergency lights today)
- Light colour: currently the raycaster outputs scalar intensity only. Fire wants warm orange/red. Either widen the light-field texture to RGB or apply a per-source tint at emission time
- Per-source `light_z` (height above floor) — currently a global uniform; fire would feel more "ground-level" with a low Z, ceiling fixtures with high Z. Not needed for v1
- Smoke advection through flamethrower jets — already handled by existing smoke dynamics; verify it reads natural with the new short-range fire lighting

## Reference

- `cpp/src/raycaster.cpp` — the C++ raycaster (DDA marching, smoke absorption, directional accumulation)
- `renderer/lighting.py` — Python-side LightingPass that drives the raycaster
- `docs/graphics_lighting_design.md` §3 — light system architecture
- `config.toml` `[physics.fire]` and `[physics.smoke]` — existing tunables (sim, not visuals)
