# G3 Fire F0: adversarial critique, 2026-10-23

Reviewed: `docs/g3_fire_design_2026-10-23.md` at `154d5f1`, checked against the code at `cfcba40`.
Evidence used: the code, `config.toml`, the fire-smoke-tuning skill, and one headless
`tools/fire_smoke_probe.py --seconds 40` run on the main checkout's build (same commit).
Erik's §1 decisions are not relitigated. The design honours them: emission-only drawing, the A/B toggle, a 2D column field that can be tilted later, and the scope order.

**Verdict: REVISE.** Three blockers sit in the core of the model:
- the source law as written draws nothing;
- the projection removes the flame's driving force;
- the composite move undoes a look ruling Erik made the day before.

The rest are MAJOR doc fixes.

---

## BLOCKERS

### B1. The source law gives a flame with zero emission. Its geometry also fits only one of the four fuels.

**Claim.** §3.3 seeds `T_vfx` from "the adjacent gas cell's game temperature". It injects at "open faces of burning tiles", meaning faces whose neighbour does not block.

**Evidence: the temperature.**
- Probe run on `fire_smoke_studio`, tile (25,22):
  - burning tile: 1331 / 1212 / 1126 / 1061 K at 5 / 10 / 15 / 20 s;
  - hottest gas nearby: 457 / 736 / 761 / 702 K.
- The glow floor is `kelvin_glow_min = 800` (`config.toml:1476`). The intensity is 0 below it (`blackbody.py:248-253`).
- The fire-smoke-tuning skill says the same: gas "peaks at ~740 K, below the ~800 K glow floor" (`SKILL.md:65-66`).
- So `BlackbodyRamp(T_vfx)·E(R)` is 0 everywhere. "Never hotter than the sim" then guarantees the flame stays invisible.

**Evidence: the geometry.**
- The studio's fuel is furniture (`levels/fire_smoke_studio/level.toml:3-9`). Furniture is *permeable*: gas seeps past a crate, and it is not `solid` (`gamemap.py:1451-1455`).
- Kindling and foliage behave the same way. Of the flammable materials, only wood is `solid`.
- The look-shot fires go further. `_inject_fire` force-flammables a 3×3 patch of **air** (`tools/lighting_demo.py:900-913`).
- In the sim, "the heat and the soot land at the FIRE", i.e. on the burning tile itself (`physics_runner.py:1232-1235`).
- An open-face-only source therefore draws a ring around the fuel with a dark centre. Seen from straight above that is backwards: the tallest, thickest column of flame stands over the fuel.

**Fix.** Write the source law per material class:
- **Permeable fuel and air fires:** inject R over the burning tile's own area.
- **Solid fuel (wood):** inject in a thin strip in the open neighbour cell along each open face.
- **Source temperature, both cases:** `T_src` = **the burning tile's own `temperature`**. That is the value the sweep emits `a·L°[T]` from and the heat layer glows with, so colour and room light still agree.
- **Restated invariant:** "`T_vfx` ≤ the `T` of the burning tile that fed it."

Tell Erik what this means for the look: burning tiles run at ~900–1330 K, so the flames will be deep orange-red. There will be no yellow-white roots unless the sim's tile temperature rises (#68).

### B2. §3.5 brings back the "fire went dark under its own smoke" look Erik ruled out on 2026-10-22, and it breaks G2's rule for hot gas.

**Claim.** "The ceiling soot's normal alpha composite then hides the flame by its own transmittance … G2's `glow_through_fs` weight was needed only because the heat layer was drawn after the resolve."

**Evidence.**
- `glow_through_ceiling = 0.2` (`config.toml:1700-1701`). It was set from 1.0 to 0.2 with Erik's words: "the fire's own smoke closed over it and it went dark". This is a look ruling. It is not left over from display-space drawing.
- Drawing under the ceiling's premultiplied composite (`gas_relief.py:718`) is exactly `glow_through_ceiling = 1.0`, the setting Erik rejected.
- It gets worse here: soot lands *on* the burning tile (`physics_runner.py:1235`), so the densest soot sits exactly over the flame.
- G2 also deliberately leaves **hot-gas** glow undimmed: "an emitting slab is not hidden by its own depth" (`overlays.py:344-352`, `gas_relief.py:384-408`, and the CLAUDE.md Smoke-cast-product row, "hot gas is #95's").
- §3.5 moves *all* heat glow, gas included, under the soot. An explosion fireball would then be hidden by its own soot.

**Fix.** Keep the one-tone-map retirement, but keep the weight:
- **Body glow and flame:** draw before the units. Add a compensation term after the ceiling pass: `glow·α_ceil·(1 − glow_through_ceiling)·below`, read from the cast product. That is G2's shader moved before the resolve.
- **Gas glow:** draw after the ceiling, undimmed (or as emission inside the relief medium: `B(T)·(1−e^{−τ})`).
- If the design really wants physical occlusion (1.0) now that the flame has shape, Erik has to rule on it. Give him an A/B shot of `fire_thick_smoke` and `fss_overview` at 1.0 vs 0.2.

### B3. The projection deletes the "outward kick", so the F2 gate ("tongues leaving the open faces") cannot pass honestly.

**Claim.** Step 2 injects "an outward velocity kick ∝ I". Step 5 then projects to zero divergence.

**Evidence.**
- A radial kick around a source is almost pure gradient. The Helmholtz projection removes exactly that part.
- Only the asymmetric residue survives, plus whatever divergence 20 un-converged Jacobi passes leave behind. That makes `n_jacobi` a look dial.
- Top-down there is no buoyancy (§3.3) to make vorticity, so the curl noise would be the only real structure source. The arc would pay for a fluid solver and get curl-noise advection.
- The physics in the plane is also mixed:
  - at floor level a fire *entrains* air inward;
  - at flame height the hot gas *expands*;
  - the plume leaves the plane upward, which looks like a sink.

**Fix.** Make the source a **prescribed divergence**: solve `∇²p = ∇·u* − s(I)`.
- Thermal expansion of the burning gas (a real ×4–5 at 1300 K) then survives as outward flow.
- This needs an **open (p = 0) boundary at the active region's edge**. Otherwise the Neumann problem in a closed room has no solution for net `Σs ≠ 0`.
- Choose the sign deliberately (expansion vs entrainment) and write it down.
- Prove it in a numpy prototype *before* the F1 GPU substrate: a 64² grid with one burning tile, showing tongues.

---

## MAJOR

### M1. The clock: frame `dt` breaks pause, slow motion, replays and look shots.

**Evidence.**
- The game is mostly paused (WEGO planning). It ticks through an accumulator scaled by `sim_speed` (`main.py:595-615`).
- Every other render animation runs on the sim tick and freezes on pause: the smoke detail, the relief, the speckle, prop sway (`advected_noise.py:13-18`, `game_renderer.py:431-432`).
- Look shots run **one sim tick per rendered frame** (`look_shots.toml:14-15`). With `dt = 1/60` in shot mode, the flame runs 2.5× slower against the sim in shots than in the game. The judged PNG would not be the played look.
- The flame edge noise is said to use "the one noise clock". That clock steps at 24 Hz, so 24 Hz-stepped noise would sit on a flame moving at 60 fps.

**Fix.**
- Make VFX time = sim time + the sub-tick fraction (`tick_accum / sim_time_per_tick`), integrated in fixed internal substeps (e.g. 1/120 s).
- Shots then run 2.5 substep-frames per tick, the same as the game.
- Ask Erik whether the flame keeps burning while paused (a feel decision). Keep the same rule for the noise clock.

### M2. Heat layer to linear HDR: G0 measured the hue loss, and the doc offers no mechanism to avoid it.

**Evidence.**
- G0 measured chroma falling from 0.219 to 0.166 on `fire_plume`, and an exact conversion still faded it to 0.179 (`graphics_direction_2026-10-18.md:545-556`).
- The doc only restates the gate.
- Separately, `enabled = false` ("today's look") stops being true from F3 on, because the heat layer moves regardless of the toggle.

**Fix.** Composite by **Kirchhoff**:
- **Flame and hot gas:** premultiplied *over*, `out = ε·B(T) + (1−ε)·dst`, with `ε = E(R)` (A: the mask, B: `1−e^{−κR}`). An opaque emitter then *replaces* the lit floor instead of adding to it, which is what G0 lost. It is still emission-only (decision 2).
- **Body glow** stays additive.
- Check in a metric (mean chroma over a fire mask, bright-area radius at a fixed threshold) in `look_shots.py`, with G0's ±4 % as the threshold.
- Redefine the toggle: `enabled = false` means "linear heat, no flame".

### M3. Deleting the display path breaks the relief's own-light cut.

**Evidence.**
- `GameRenderer._heat_brightness` feeds `glow_below_fraction` and the relief's `below_glow_field` from `pack_emissive_rgba`'s display alpha (`game_renderer.py:977-1004`, `gas_relief.py:373-381, 637-659`).
- `below_glow_field` clips to [0,1]. Linear intensities run up to 64 (`config.toml:1488`), so every glowing tile would saturate to 1.
- `aces_curve` may not be called outside `blackbody.py` (`test_one_tonemap.py:79-100`).

**Fix.** Define the new [0,1] glow weight explicitly, e.g. `1 − exp(−I/I₀)` on the linear intensity, in `blackbody.py` as a pure function. Check that the G2 shots (`smoke_over_squad`, `fire_thin_smoke`) are unchanged.

### M4. The flame's brightness is counted twice.

**Evidence.**
- The burning tile already glows at `B(T_tile)` through the heat layer.
- The flame adds `B(T_vfx)·E` over the same tile and around it.
- The sweep lights the room from one tile's emission. A flame 2–3 tiles in luminous area then looks brighter than the light it casts. Mode A makes this worst.

**Fix.**
- Partition: the flame carries the burning tile's emission, and the heat layer keeps non-burning bodies plus an ember-bed share weighted by `(1 − I)`.
- Add a property test: the flame's integrated radiance per burning tile stays within a factor of that tile's `B(T)` × its area.

### M5. A nearest-sampled 256-entry LUT bands a continuous flame.

**Evidence.**
- The LUT has 256 entries over 800–10 000 K, about 36 K per step, sampled at the nearest index (`blackbody.py:231-233, 284-289`).
- At 1000 K one step changes intensity from 0.16 to 0.22, a 39 % jump. That was harmless on per-tile bilinear data but contours a continuous `T_vfx`.
- "Equals sample for sample" is also impossible in an fp16 texture.
- `lut_texture()` would make `blackbody.py` import pyray, but the module is pyray-free by design (`blackbody.py:63-64`).

**Fix.**
- Linear-filter the GPU LUT and/or raise `lut_size`. The sim's L° table comes from `emission_at_kelvin` (`blackbody.py:265-271`), not the LUT, so no digest moves.
- Export the array from `blackbody.py` and upload it in `fire_vfx.py`.
- Test: LUT texels equal the ramp's entries within fp16/fp32 tolerance, and the shader index uses the same `[physics.temperature_scale]` map. Add Temperature scale to Systems (a).

### M6. The budget leaves out the dominant costs.

**Evidence.**
- The estimate is bandwidth only.
- About 28–32 fragment passes per frame each pay Python/cffi plus rlgl setup: texture mode, shader mode, uniforms, sampler binds, flush. The heat overlay even calls `get_shader_location` on every draw (`overlays.py:458-465`).
- That is easily 1–2 ms of CPU per frame before any GPU work.
- The **draw** into the 48 px/tile world RT is per world pixel. A whole-map fire on 128×256 is 75 MP, which the relief avoids only by clipping to the view (`game_renderer.py:1025-1034`).

**Fix.**
- Clip the draw to view ∩ active region.
- Warm-start `p` from the last frame (cuts `n_jacobi`).
- Cache uniform locations.
- Measure CPU submit time and GPU time (glFinish-bracketed in `--perf`) separately.
- Correct "half-res halves the solve": in 2D it quarters it.

### M7. The active-region scissor on a world-anchored grid leaves stale state.

**Evidence.**
- When the box shrinks or moves, texels outside it keep their last R/T/u. The full-map draw then shows frozen flames, or a clipped draw cuts flames at a hard rectangle.
- Two distant fires make one map-spanning box.
- The margin must cover the travel speed: tamed wind saturates at 1.4 tiles/tick = 33.6 tiles/s (`gas_detail.py:90`).

**Fix.**
- Use one rect per fire cluster (or a tile-occupancy mask).
- Clear on shrink.
- Use the same rects for the sim and the draw.
- Clamp or cut sources at the boundary.
- Use the B3 open boundary at the rect edge.

### M8. The ping-pong helper needs its traps written into the design.

**Evidence and fix.**
- **Blending:** raylib's default `BLEND_ALPHA` is active inside `begin_texture_mode`. Any field stored in alpha, or any shader not writing `a = 1`, is silently blended. Use a custom `ONE, ZERO` blend for the sim passes.
- **Y flip:** an RT read with a positive source height flips every pass (`world_composite.py:36-40`), and a flipped velocity also needs `v_y` negated.
- **Depth buffers:** `load_render_texture_rgba16f` attaches a depth buffer (`core.py:184-187`), which is wasted VRAM here. Use colour-only FBOs.
- **Pressure precision:** fp16 pressure is marginal under Jacobi. Consider R32F.
- **Gate:** an identity shader run N times (N odd and even) leaves the field bit-identical, checked by GL readback in a subprocess harness.

### M9. The flicker analysis (§6.3) is wrong about option C and misses a cleaner option.

**Evidence.**
- The sweep's light channels "move no heat integer (heat bit-identical with light on vs off)" (`radiation_sweep.h:79-85`).
- `light_q` is not digested (`light_field.py:31-33`).
- So modulating only the *light-channel* emission of burning cells does **not** make heat flicker or move digests and the golden.

**Missing option D.** Pass a per-tick, per-burning-cell light-emission multiplier from the VFX's local R into the sweep as a render-state input, the way the cone rows, the cursor lamp and the muzzle flash already enter (`config.toml` `[render.muzzle_flash]`: "render state, never sim state, no digest").
- The room light then flickers at the fire only, and lamps are untouched.
- It steps at 24 Hz, which is above the Nyquist rate for 1–10 Hz flame flicker.
- It is C++ work in the sweep's input. It still belongs in Erik's ruling, but Erik should rule on the correct menu: A, B, D.
- Option C as written ("heat flickers, golden moves") is only one way to do it, and not the cheap one.

### M10. The shimmer pass cannot read the scene RT while writing it.

**Evidence.** §6.2's separate pass needs a full copy of the world-sized RGBA16F target: 129 MB on the playground (`world_composite.py:32-34`). The "reserved post-FX slot" is a docstring sentence (`world_composite.py:29`), not existing code.

**Fix.**
- Make shimmer a UV-offset term **inside `world_resolve.fs`**: no copy, no extra pass.
- Update the draft rule: "every screen-space distortion is a term in the resolve".
- Drive it from the flame's `T_vfx` and/or hot gas, clipped to hot regions.

### M11. Emitting embers from "the hottest flame texels" means a GPU-to-CPU readback stall every frame.

**Fix.**
- Emit from the CPU source list (burning tiles, I, `T_tile`, seeded RNG) and advect with the tamed wind plus curl noise on the CPU, as §6.1 already does.
- Draw in one batched call, not one Python draw per particle.
- Say where embers draw relative to the ceiling soot.

### M12. Resolution: a lone burning tile is 4×4 velocity cells, which gives an orange blob.

**Evidence.**
- At `M_u = 4` the `fss_lone` shot has no room for tongues, and the laptop tier (`M_u = 2`) has no room for any structure.
- The proposed edge noise is the smoke **billow** (convex Perlin-Worley domes; CLAUDE.md Billow-noise row). It would make the flame edge read as orange cauliflower smoke.

**Fix.**
- Run `M_u = M` inside the active rect (cheap at studio size per §4's own arithmetic).
- Add a flow-stretched or anisotropic flame-noise function to `noise_bake.py` (the rule allows a new shape there).
- Make `fss_lone` an explicit gate shot.

---

## MINOR

1. **Factual (§2):**
   - `hover_readout.py:161` also reads `gmap.fire`.
   - Today the heat layer is drawn **above** the 3D units (`game_renderer.py:835-836`); moving it below them is a look change, so say so.
   - Light uploads also happen on 4 Hz paused relights (`main.py:621-628`).
2. **Floor haze:** `glow_through_floor = 0.3` (floor haze dims the body glow) is lost if heat draws after the floor layer. Draw the body glow before the floor gas, the flame after it.
3. **Advection limiter:** BFECC/MacCormack needs Selle's min/max limiter. Without it, advection makes new extrema (T above the source, R < 0), and the T-bound test, which only checks extraction, would miss it.
4. **Premultiplied temperature:** advect `R·T` (premultiplied) rather than T alone, or T at a vanishing edge carries hot values into empty air.
5. **Wind:** reuse the `tame_wind` array the relief already computes per frame (`game_renderer.py:502-505`). Do not call it a third time with possibly different `v_ref`/`v_sens` (props use the defaults).
6. **Systems (a) is missing:**
   - Temperature scale.
   - The lit3d `smokeCoverAt` seam, if any glow-through survives (B2).
   - The exact text edits to the CLAUDE.md "One tone-map" row (`CLAUDE.md:124`, "never converted to linear") and the "Smoke cast product" row (`:131`).
   - `frame_lights.py`/sweep inputs, if flicker option D is chosen.
7. **Particle rule:** scope it so the existing `_effects` queue is named as legacy, not a second system.
8. **§8 render-only test:** GL tests are scripts, not collected (`test_renderer_smoke.py`). Enforce structurally instead: hand `fire_vfx` `writeable=False` views, add a lint that it never assigns to `gmap`, and run a GL subprocess harness for the digest A/B. The "one tone-map, no exception" test is grep-based; add a draw-order test with a recording pyray stub.
9. **§8 missing tests:**
   - no fire → zero VFX draws;
   - R = 0 inside solids after every frame;
   - `max T_vfx ≤ max T_src` and `R ≥ 0` on a numpy reference loop (the spec twin), with GL parity checked by readback;
   - a shot pair where A and B actually differ (the steep `((K−800)/500)²` falloff may make T, not R, set the visible edge in both).
10. **Config naming:** F5 vs Ctrl+R is inconsistent between §5 and §7. Changing `m`/`m_u` live means reallocating the RTs.
11. **Props:** future foliage props draw in the 3D pass *after* the flame and would cover their own fire. Note it for #60.
12. **Smoke steps visibly:** with a 60 fps flame, the tick-stepped smoke relief may look choppy beside it. The M1 sub-tick clock fixes both.
13. **Vacuum:** keep the vacuum gate on sources (`game_renderer.py:568`).

---

## Ranked summary

| # | Severity | One line |
|---|---|---|
| B1 | BLOCKER | `T_src` from adjacent gas (≤ 761 K) < 800 K glow floor → invisible flame; the open-face geometry is wrong for permeable fuel and the studio's air fires |
| B2 | BLOCKER | The composite under the ceiling soot = `glow_through_ceiling` 1.0, which Erik rejected on 2026-10-22; it also hides hot-gas glow against G2's rule |
| B3 | BLOCKER | The projection removes the outward kick; it needs a prescribed-divergence source plus an open boundary, proven in numpy first |
| M1 | MAJOR | Frame-dt clock: pause, `sim_speed`, shots (1 tick per frame) and the tick-locked noise all disagree |
| M2 | MAJOR | Linear heat has no mechanism against G0's measured hue loss; use the Kirchhoff over-operator, a checked-in metric, and a redefined toggle |
| M3 | MAJOR | `_heat_brightness` and the own-light cut depend on the deleted display pack |
| M4 | MAJOR | The flame plus the heat layer count the burning tile's emission twice |
| M5 | MAJOR | Nearest 256-entry LUT bands the flame; keep `blackbody.py` pyray-free |
| M6 | MAJOR | The budget ignores per-pass CPU overhead and the world-RT draw |
| M7 | MAJOR | The scissored world grid leaves stale state; use per-cluster rects |
| M8 | MAJOR | Ping-pong blend, Y flip, depth buffers and fp16 pressure need specifying |
| M9 | MAJOR | Flicker option C is mis-analysed; option D (light-only emission, not digested) is missing |
| M10 | MAJOR | Shimmer needs a scene copy; fold it into the resolve |
| M11 | MAJOR | Embers from GPU texels need a readback stall; emit from the CPU source list |
| M12 | MAJOR | `M_u = 4` makes a lone tile a blob; billow noise makes cauliflower edges |

**Overall: REVISE.** B1–B3 change the source law, the projection and the composite order. B2 needs Erik's ruling (or the keep-0.2 design), and so do M1 (flame while paused) and M9 (the corrected flicker menu). The rest are doc edits.
